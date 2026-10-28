defmodule ReqLLM.Providers.Anthropic.ResponseBuilder do
  @moduledoc """
  Anthropic-specific ResponseBuilder implementation.

  Handles Anthropic's specific requirements:
  - Content blocks must be non-empty when tool_calls are present
  - Maps `tool_use` finish reason to `:tool_calls`

  This fixes bug #269 where streaming tool-call-only responses
  produced empty content blocks that Anthropic's API rejected.
  """

  @behaviour ReqLLM.Provider.ResponseBuilder

  alias ReqLLM.Message
  alias ReqLLM.Message.ContentPart
  alias ReqLLM.Provider.Defaults.ResponseBuilder, as: DefaultBuilder
  alias ReqLLM.StreamChunk

  @impl true
  def build_response(chunks, metadata, opts) do
    with {:ok, response} <- DefaultBuilder.build_response(chunks, metadata, opts) do
      {:ok, response |> put_raw_content(chunks) |> ensure_non_empty_content()}
    end
  end

  @doc false
  @spec build_buffered_response([ReqLLM.StreamChunk.t()], map(), keyword()) ::
          {:ok, ReqLLM.Response.t()} | {:error, term()}
  def build_buffered_response(chunks, metadata, opts) do
    with {:ok, response} <- DefaultBuilder.build_buffered_response(chunks, metadata, opts) do
      {:ok, preserve_empty_message(response, chunks, opts)}
    end
  end

  defp ensure_non_empty_content(%{message: %{tool_calls: tc, content: []}} = response)
       when is_list(tc) and tc != [] do
    content = [ContentPart.text("")]
    put_in(response.message.content, content)
  end

  defp ensure_non_empty_content(response), do: response

  # Streamed counterpart of the buffered `anthropic_content` message metadata:
  # the decoder announces each block's start and emits it whole once finished;
  # they are reassembled in index order. A started block that never finished,
  # or one the decoder marked `:incomplete`, means the turn can't be replayed
  # faithfully, so the message keeps only its normalized form.
  defp put_raw_content(%{message: %Message{} = message} = response, chunks) do
    case raw_content(chunks) do
      nil ->
        response

      content ->
        updated = %{message | metadata: Map.put(message.metadata, :anthropic_content, content)}
        %{response | message: updated, context: replace_last(response.context, message, updated)}
    end
  end

  defp put_raw_content(response, _chunks), do: response

  defp raw_content(chunks) do
    started =
      for %StreamChunk{type: :meta, metadata: %{anthropic_content_block_started: index}} <-
            chunks,
          uniq: true,
          do: index

    blocks =
      for %StreamChunk{type: :meta, metadata: %{anthropic_content_block: block}} <- chunks,
          do: block

    cond do
      blocks == [] -> nil
      Enum.sort(started) != blocks |> Enum.map(& &1.index) |> Enum.sort() -> nil
      Enum.any?(blocks, &(&1.block == :incomplete)) -> nil
      true -> blocks |> Enum.sort_by(& &1.index) |> Enum.map(& &1.block)
    end
  end

  defp replace_last(%ReqLLM.Context{messages: messages} = context, old, new) do
    if List.last(messages) == old,
      do: %{context | messages: List.replace_at(messages, -1, new)},
      else: context
  end

  defp replace_last(context, _old, _new), do: context

  defp preserve_empty_message(response, chunks, opts) do
    if Enum.any?(chunks, &message_chunk?/1) do
      response
    else
      %{response | message: nil, context: Keyword.fetch!(opts, :context)}
    end
  end

  defp message_chunk?(%ReqLLM.StreamChunk{type: type})
       when type in [:content, :content_part, :thinking, :tool_call],
       do: true

  defp message_chunk?(_chunk), do: false
end
