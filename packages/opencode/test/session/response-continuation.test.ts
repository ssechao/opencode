import { describe, expect, test } from "bun:test"
import { Effect } from "effect"
import type { Provider } from "@/provider/provider"
import type { SessionV1 } from "@opencode-ai/core/v1/session"
import { ResponseContinuation } from "@/session/llm/response-continuation"
import { LLMRequestPrep } from "@/session/llm/request"
import { ProviderTransform } from "@/provider/transform"
import { createOpenAI } from "@ai-sdk/openai"
import { APICallError, streamText, type ModelMessage } from "ai"

const sessionID = "ses_continuation" as any

function model(input?: { store?: boolean; providerID?: string; modelID?: string; npm?: string }) {
  return {
    id: input?.modelID ?? "claude-opus-5",
    providerID: input?.providerID ?? "weytop-wrapper",
    api: {
      id: input?.modelID ?? "claude-opus-5",
      url: "http://wrapper.test/v1",
      npm: input?.npm ?? "@ai-sdk/openai",
    },
    options: input?.store === false ? {} : { responsesContinuation: true },
    headers: {},
    capabilities: {
      temperature: false,
      reasoning: true,
      attachment: true,
      toolcall: true,
      input: { text: true, audio: false, image: true, video: false, pdf: true },
      output: { text: true, audio: false, image: false, video: false, pdf: false },
      interleaved: false,
    },
    cost: { input: 0, output: 0, cache: { read: 0, write: 0 } },
    limit: { context: 1_000_000, output: 128_000 },
    status: "active",
    name: "Claude Opus 5",
    family: "claude",
    variants: {},
  } as unknown as Provider.Model
}

function user(id: string, text: string, created: number): SessionV1.WithParts {
  return {
    info: {
      id,
      sessionID,
      role: "user",
      time: { created },
      agent: "build",
      model: { providerID: "weytop-wrapper", modelID: "claude-opus-5" },
    },
    parts: [{ id: `prt_${id}`, messageID: id, sessionID, type: "text", text }],
  } as unknown as SessionV1.WithParts
}

function assistant(input: {
  id: string
  parentID: string
  created: number
  responseId?: string
  providerID?: string
  modelID?: string
  tool?: boolean
  finish?: "stop" | "tool-calls" | "unknown"
}): SessionV1.WithParts {
  const providerID = input.providerID ?? "weytop-wrapper"
  const modelID = input.modelID ?? "claude-opus-5"
  return {
    info: {
      id: input.id,
      parentID: input.parentID,
      sessionID,
      role: "assistant",
      mode: "build",
      agent: "build",
      path: { cwd: "/tmp", root: "/tmp" },
      cost: 0,
      tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
      providerID,
      modelID,
      time: { created: input.created, completed: input.created + 1 },
      finish: input.finish ?? (input.tool ? "tool-calls" : "stop"),
    },
    parts: [
      ...(input.tool
        ? [
            {
              id: `prt_tool_${input.id}`,
              messageID: input.id,
              sessionID,
              type: "tool" as const,
              callID: "call_1",
              tool: "bash",
              state: {
                status: "completed" as const,
                input: { command: "pwd" },
                output: "/tmp",
                title: "pwd",
                metadata: {},
                time: { start: input.created, end: input.created + 1 },
              },
            },
          ]
        : [
            {
              id: `prt_text_${input.id}`,
              messageID: input.id,
              sessionID,
              type: "text" as const,
              text: "answer",
            },
          ]),
      {
        id: `prt_finish_${input.id}`,
        messageID: input.id,
        sessionID,
        type: "step-finish" as const,
        reason: input.finish ?? (input.tool ? "tool-calls" : "stop"),
        cost: 0,
        tokens: { input: 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
        providerMetadata: input.responseId ? { openai: { responseId: input.responseId } } : undefined,
      },
    ],
  } as unknown as SessionV1.WithParts
}

describe("Responses continuation", () => {
  test("serializes only standard continuation fields on the Responses wire", async () => {
    const bodies: Record<string, unknown>[] = []
    const provider = createOpenAI({
      apiKey: "test-key",
      baseURL: "http://wrapper.test/v1",
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(JSON.parse(String(init?.body)))
        const events = [
          {
            type: "response.created",
            sequence_number: 1,
            response: { id: "resp_2", created_at: 1, model: "claude-opus-5", service_tier: null },
          },
          {
            type: "response.output_item.added",
            sequence_number: 2,
            output_index: 0,
            item: { type: "message", id: "msg_1" },
          },
          {
            type: "response.output_text.delta",
            sequence_number: 3,
            item_id: "msg_1",
            delta: "ok",
            logprobs: null,
          },
          {
            type: "response.output_item.done",
            sequence_number: 4,
            output_index: 0,
            item: { type: "message", id: "msg_1" },
          },
          {
            type: "response.completed",
            sequence_number: 5,
            response: {
              incomplete_details: null,
              service_tier: null,
              usage: {
                input_tokens: 1,
                input_tokens_details: { cached_tokens: 0 },
                output_tokens: 1,
                output_tokens_details: { reasoning_tokens: 0 },
              },
            },
          },
        ]
        const body = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`
        return new Response(body, { headers: { "content-type": "text/event-stream" } })
      }) as any,
    })

    const result = streamText({
      model: provider.responses("claude-opus-5"),
      messages: [{ role: "user", content: "second" }],
      providerOptions: {
        openai: {
          store: true,
          previousResponseId: "resp_1",
          instructions: "SYSTEM",
        },
      },
    })
    const stream = await Array.fromAsync(result.fullStream)
    expect(
      stream
        .filter((event) => event.type === "text-delta")
        .map((event) => event.text)
        .join(""),
    ).toBe("ok")
    expect(stream.find((event) => event.type === "finish-step")?.providerMetadata).toMatchObject({
      openai: { responseId: "resp_2" },
    })

    expect(bodies).toHaveLength(1)
    expect(bodies[0]).toMatchObject({
      model: "claude-opus-5",
      store: true,
      previous_response_id: "resp_1",
      instructions: "SYSTEM",
      input: [{ role: "user", content: [{ type: "input_text", text: "second" }] }],
    })
    for (const key of Object.keys(bodies[0])) {
      expect(key.startsWith("x-")).toBe(false)
    }
  })

  test("maps continuation to standard OpenAI Responses options without replaying system messages", async () => {
    const mdl = model()
    const result = await Effect.runPromise(
      LLMRequestPrep.prepare({
        user: {
          id: "msg_user_2",
          sessionID,
          role: "user",
          time: { created: 3 },
          agent: "build",
          model: { providerID: mdl.providerID, modelID: mdl.id },
        } as SessionV1.User,
        sessionID,
        model: mdl,
        agent: {
          name: "build",
          mode: "primary",
          prompt: "SYSTEM A",
          options: {},
          permission: [],
        } as any,
        system: ["SYSTEM B"],
        messages: [{ role: "user", content: "delta" }],
        tools: {},
        provider: { id: "weytop-wrapper", options: {} } as any,
        auth: undefined,
        plugin: {
          trigger: (_name: string, _input: unknown, output: unknown) => Effect.succeed(output),
        } as any,
        flags: { outputTokenMax: 128_000, client: "test" } as any,
        isWorkflow: false,
        responseContinuation: { previousResponseId: "resp_1" },
      }),
    )

    expect(result.messages).toEqual([{ role: "user", content: "delta" }])
    expect(result.params.options.store).toBe(true)
    expect(result.params.options.previousResponseId).toBe("resp_1")
    expect(result.params.options.instructions).toBe("SYSTEM A\nSYSTEM B")
    expect(result.params.options.responsesContinuation).toBeUndefined()
    expect(result.messageTransformOptions.store).toBe(true)
    expect(Object.keys(result.headers).map((key) => key.toLowerCase())).not.toContain("x-session-id")
    expect(Object.keys(result.headers).map((key) => key.toLowerCase())).not.toContain("x-session-affinity")
    expect(Object.keys(result.headers).map((key) => key.toLowerCase())).not.toContain("x-parent-session-id")
  })

  test("bootstraps a stored chain without replaying stale item references", async () => {
    const mdl = model()
    const result = await Effect.runPromise(
      LLMRequestPrep.prepare({
        user: {
          id: "msg_user_1",
          sessionID,
          role: "user",
          time: { created: 1 },
          agent: "build",
          model: { providerID: mdl.providerID, modelID: mdl.id },
        } as SessionV1.User,
        sessionID,
        model: mdl,
        agent: { name: "build", mode: "primary", prompt: "SYSTEM", options: {}, permission: [] } as any,
        system: [],
        messages: [{ role: "user", content: "full history" }],
        tools: {},
        provider: { id: "weytop-wrapper", options: {} } as any,
        auth: undefined,
        plugin: {
          trigger: (_name: string, _input: unknown, output: unknown) => Effect.succeed(output),
        } as any,
        flags: { outputTokenMax: 128_000, client: "test" } as any,
        isWorkflow: false,
        responseContinuation: {},
      }),
    )

    expect(result.params.options.store).toBe(true)
    expect(result.params.options.previousResponseId).toBeUndefined()
    expect(result.messageTransformOptions.store).toBe(false)

    const transformed = ProviderTransform.message(
      [
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "old answer",
              providerOptions: { openai: { itemId: "msg_from_stateless_history" } },
            },
          ],
        },
      ] satisfies ModelMessage[],
      mdl,
      result.messageTransformOptions,
    )
    const content = transformed[0]?.content
    if (!Array.isArray(content) || content[0]?.type !== "text") throw new Error("expected an assistant text part")
    expect(content[0].providerOptions?.openai?.itemId).toBeUndefined()
  })

  test("keeps the existing full-history behavior unless explicitly enabled", async () => {
    const messages = [
      user("msg_user_1", "first", 1),
      assistant({
        id: "msg_assistant_1",
        parentID: "msg_user_1",
        created: 2,
        responseId: "resp_1",
      }),
      user("msg_user_2", "second", 3),
    ]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model({ store: false }) }))

    expect(result.enabled).toBe(false)
    expect(result.previousResponseId).toBeUndefined()
    expect(result.messages.map((message) => message.role)).toEqual(["user", "assistant", "user"])
  })

  test("continues a stored response with only the new user delta", async () => {
    const messages = [
      user("msg_user_1", "first", 1),
      assistant({
        id: "msg_assistant_1",
        parentID: "msg_user_1",
        created: 2,
        responseId: "resp_1",
      }),
      user("msg_user_2", "second", 3),
    ]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model() }))

    expect(result.enabled).toBe(true)
    expect(result.previousResponseId).toBe("resp_1")
    expect(result.recoveryHistory).toBe(messages)
    expect(result.messages).toEqual([{ role: "user", content: [{ type: "text", text: "second" }] }])
  })

  test("never reuses a response id from an unknown finish", async () => {
    const messages = [
      user("msg_user_1", "first", 1),
      assistant({
        id: "msg_assistant_1",
        parentID: "msg_user_1",
        created: 2,
        responseId: "resp_missing",
        finish: "unknown",
      }),
      user("msg_user_2", "retry", 3),
    ]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model() }))

    expect(result.previousResponseId).toBeUndefined()
    expect(result.recoveryHistory).toBeUndefined()
    expect(result.messages.map((message) => message.role)).toEqual(["user", "assistant", "user"])
  })

  test("recognizes only a structured missing stored Response error", () => {
    const error = new APICallError({
      message: "No response found",
      url: "http://wrapper.test/v1/responses",
      requestBodyValues: {},
      statusCode: 404,
      responseHeaders: {},
      responseBody: JSON.stringify({ error: { code: "resource_not_found" } }),
      isRetryable: false,
    })
    expect(ResponseContinuation.missingStoredResponse(error)).toBe(true)
    expect(
      ResponseContinuation.missingStoredResponse(
        new APICallError({
          message: "other 404",
          url: "http://wrapper.test/v1/responses",
          requestBodyValues: {},
          statusCode: 404,
          responseHeaders: {},
          responseBody: JSON.stringify({ error: { code: "other" } }),
          isRetryable: false,
        }),
      ),
    ).toBe(false)
  })

  test("continues a tool response with only its locally executed tool output", async () => {
    const messages = [
      user("msg_user_1", "run pwd", 1),
      assistant({
        id: "msg_assistant_1",
        parentID: "msg_user_1",
        created: 2,
        responseId: "resp_tool",
        tool: true,
      }),
    ]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model() }))

    expect(result.previousResponseId).toBe("resp_tool")
    expect(result.messages).toHaveLength(1)
    expect(result.messages[0]).toMatchObject({
      role: "tool",
      content: [{ type: "tool-result", toolCallId: "call_1", toolName: "bash" }],
    })
  })

  test("starts a new stored chain after a model change", async () => {
    const messages = [
      user("msg_user_1", "first", 1),
      assistant({
        id: "msg_assistant_1",
        parentID: "msg_user_1",
        created: 2,
        responseId: "resp_1",
        modelID: "claude-sonnet-5",
      }),
      user("msg_user_2", "second", 3),
    ]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model() }))

    expect(result.previousResponseId).toBeUndefined()
    expect(result.messages.map((message) => message.role)).toEqual(["user", "assistant", "user"])
  })

  test("does not continue from an incomplete assistant response", async () => {
    const prior = assistant({
      id: "msg_assistant_1",
      parentID: "msg_user_1",
      created: 2,
      responseId: "resp_partial",
    })
    if (prior.info.role !== "assistant") throw new Error("fixture must be an assistant message")
    prior.info.time.completed = undefined
    const messages = [user("msg_user_1", "first", 1), prior, user("msg_user_2", "retry", 3)]
    const result = await Effect.runPromise(ResponseContinuation.prepare({ messages, model: model() }))

    expect(result.previousResponseId).toBeUndefined()
    expect(result.messages.map((message) => message.role)).toEqual(["user", "assistant", "user"])
  })

  test("starts a fresh stored chain when an automatic retry has no continuation delta", async () => {
    const prior = assistant({
      id: "msg_assistant_1",
      parentID: "msg_user_1",
      created: 2,
      responseId: "resp_output_only",
    })
    if (prior.info.role !== "assistant") throw new Error("fixture must be an assistant message")
    prior.info.finish = "unknown"
    const result = await Effect.runPromise(
      ResponseContinuation.prepare({ messages: [user("msg_user_1", "first", 1), prior], model: model() }),
    )

    expect(result.enabled).toBe(true)
    expect(result.previousResponseId).toBeUndefined()
    expect(result.messages.map((message) => message.role)).toEqual(["user", "assistant"])
  })

  test("starts a new stored chain immediately after compaction", async () => {
    const first = user("msg_user_1", "first", 1)
    const old = assistant({
      id: "msg_assistant_1",
      parentID: "msg_user_1",
      created: 2,
      responseId: "resp_old",
    })
    const compact = user("msg_user_compact", "compact", 3)
    compact.parts.push({
      id: "prt_compaction",
      messageID: compact.info.id,
      sessionID,
      type: "compaction",
      auto: true,
    } as unknown as SessionV1.Part)
    const next = user("msg_user_2", "continue", 5)
    const result = await Effect.runPromise(
      ResponseContinuation.prepare({
        messages: [compact, old, first, next],
        model: model(),
      }),
    )

    expect(result.previousResponseId).toBeUndefined()
    expect(result.messages.length).toBeGreaterThan(1)
  })
})
