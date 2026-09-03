import { SessionV1 } from "@opencode-ai/core/v1/session"
import { Effect } from "effect"
import type { ModelMessage } from "ai"
import type { Provider } from "@/provider/provider"
import { MessageV2 } from "../message-v2"

const RESPONSE_ID = /^[A-Za-z0-9_-]{1,256}$/

export type Prepared = {
  readonly enabled: boolean
  readonly previousResponseId?: string
  readonly messages: ModelMessage[]
}

export function enabled(model: Provider.Model) {
  return model.api.npm === "@ai-sdk/openai" && model.options.responsesContinuation === true
}

function after(left: SessionV1.Info, right: SessionV1.Info) {
  if (left.time.created !== right.time.created) return left.time.created > right.time.created
  return left.id > right.id
}

function responseID(message: SessionV1.WithParts): string | undefined {
  if (message.info.role !== "assistant") return undefined
  const finish = message.parts.findLast((part): part is SessionV1.StepFinishPart => part.type === "step-finish")
  const openai = finish?.providerMetadata?.openai
  if (!openai || typeof openai !== "object" || Array.isArray(openai)) return undefined
  const value = openai.responseId
  if (typeof value !== "string" || !RESPONSE_ID.test(value)) return undefined
  return value
}

function latestAssistant(messages: SessionV1.WithParts[]) {
  let result: SessionV1.WithParts | undefined
  for (const message of messages) {
    if (message.info.role !== "assistant" || message.info.summary) continue
    if (!result || after(message.info, result.info)) result = message
  }
  return result
}

function latestCompaction(messages: SessionV1.WithParts[]) {
  let result: SessionV1.WithParts | undefined
  for (const message of messages) {
    if (message.info.role !== "user" || !message.parts.some((part) => part.type === "compaction")) continue
    if (!result || after(message.info, result.info)) result = message
  }
  return result
}

export const prepare = Effect.fn("ResponseContinuation.prepare")(function* (input: {
  readonly messages: SessionV1.WithParts[]
  readonly model: Provider.Model
}) {
  if (!enabled(input.model)) {
    return {
      enabled: false,
      messages: yield* MessageV2.toModelMessagesEffect(input.messages, input.model),
    } satisfies Prepared
  }

  const anchor = latestAssistant(input.messages)
  const compacted = latestCompaction(input.messages)
  const canContinue =
    anchor?.info.role === "assistant" &&
    anchor.info.providerID === input.model.providerID &&
    anchor.info.modelID === input.model.id &&
    anchor.info.time.completed !== undefined &&
    anchor.info.finish !== undefined &&
    anchor.info.error === undefined &&
    (!compacted || after(anchor.info, compacted.info))
  const previousResponseId = canContinue && anchor ? responseID(anchor) : undefined

  if (!anchor || !previousResponseId) {
    return {
      enabled: true,
      messages: yield* MessageV2.toModelMessagesEffect(input.messages, input.model),
    } satisfies Prepared
  }

  // The previous Response already owns the assistant output and function calls.
  // Only locally executed tool results from that Response plus newer messages are
  // the delta required by the standard previous_response_id contract.
  const anchorMessages = yield* MessageV2.toModelMessagesEffect([anchor], input.model)
  const tail = input.messages.filter((message) => after(message.info, anchor.info))
  const tailMessages = yield* MessageV2.toModelMessagesEffect(tail, input.model)
  const delta = [...anchorMessages.filter((message) => message.role === "tool"), ...tailMessages]
  // OpenCode deliberately retries an "unknown" finish. A Responses request
  // cannot contain zero messages, so an output-only anchor has no standard
  // delta to continue with. Rebuild a fresh stored chain for this exceptional
  // recovery path instead of fabricating a user message or failing locally.
  if (delta.length === 0) {
    return {
      enabled: true,
      messages: yield* MessageV2.toModelMessagesEffect(input.messages, input.model),
    } satisfies Prepared
  }
  return {
    enabled: true,
    previousResponseId,
    messages: delta,
  } satisfies Prepared
})

export * as ResponseContinuation from "./response-continuation"
