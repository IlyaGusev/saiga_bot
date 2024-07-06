const version = '0.4.0';

addEventListener('fetch', (event) => {
  event.respondWith(handleRequest(event.request));
});

const CLAUDE_API_KEY = ''; // Optional: default claude api key if you don't want to pass it in the request header
const CLAUDE_BASE_URL = 'https://api.anthropic.com'; // Change this if you are using a self-hosted endpoint

const stop_reason_map = {
  end_turn: 'stop',
  stop_sequence: 'stop',
  max_tokens: 'length',
};

function getAPIKey(headers) {
  const authorization = headers.authorization;
  if (authorization) {
    return authorization.split(' ')[1] || CLAUDE_API_KEY;
  }
  return CLAUDE_API_KEY;
}

function claudeToChatGPTResponse(claudeResponse) {
  const completion = claudeResponse["content"][0]["text"];
  const timestamp = Math.floor(Date.now() / 1000);
  const promptTokens = claudeResponse["usage"]["input_tokens"];
  const completionTokens = claudeResponse["usage"]["output_tokens"];
  const stopReason = claudeResponse['stop_reason'];
  const isToolUse = stopReason == "tool_use";

  const result = {
    id: claudeResponse["id"],
    created: timestamp,
    model: claudeResponse["model"],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
    choices: [
      {
        index: 0,
        finish_reason: stopReason
          ? stop_reason_map[stopReason]
          : null,
      },
    ],
  };
  var message = {
    role: 'assistant',
    content: completion,
  };
  if (isToolUse) {
    message.tool_calls = [];
    console.log(claudeResponse["content"]);
    for (var m of claudeResponse["content"]) {
      if (m["type"] == "tool_use") {
        message.tool_calls.push({
          id: m["id"],
          function: {
            name: m["name"],
            arguments: JSON.stringify(m["input"])
          }
        })
      }
    }
  }
  result.object = 'chat.completion';
  result.choices[0].message = message;
  return result;
}

async function handleRequest(request) {
  if (request.method === 'OPTIONS') {
    return handleOPTIONS();
  } else if (request.method === 'POST') {
    const headers = Object.fromEntries(request.headers);
    const apiKey = getAPIKey(headers);
    if (!apiKey) {
      return new Response('Not Allowed', {
        status: 403,
      });
    }

    const requestBody = await request.json();
    let { model, messages, temperature, stop, max_tokens, top_p, tools} = requestBody;

    let systemPrompt = "";
    let newMessages = messages;
    if (messages[0].role == "system") {
        systemPrompt = messages[0].content;
        newMessages = messages.slice(1);
    }
    messages = newMessages;

    newMessages = [];
    for (var m of messages) {
      if (m.role == "tool") {
        m.role = "user";
        const part = {
          tool_use_id: m.tool_call_id,
          type: "tool_result",
          content: m.content
        };
        if (newMessages.at(-1).content.at(-1).type == "tool_result") {
          newMessages.at(-1).content.push(part);
          continue
        }
        m.content = [part]
        delete m.tool_call_id
        delete m.name
      }
      if (m.role == "assistant" && "tool_calls" in m) {
        var content = []
        content.push({
          type: "text",
          text: "Calling tools"
        })
        for (var call of m.tool_calls) {
          content.push({
            type: "tool_use",
            id: call.id,
            name: call.function.name,
            input: JSON.parse(call.function.arguments)
          })
        } 
        newMessages.push({
          role: "assistant",
          content: content
        })
        continue;
      }
      if (m.role == "user" && Array.isArray(m.content)) {
        let fixed_parts = []
        for (var part of m.content) {
          if (part.type == "image_url") {
            const image_url = part.image_url.url;
            const base64 = image_url.replace("image/jpeg;base64,", "").replace("data:", "");
            fixed_parts.unshift({
              type: "image",
              source: {
                type: "base64",
                media_type: "image/jpeg",
                data: base64
              }
            })
          } else {
            fixed_parts.push(part)
          }
        }
        m.content = fixed_parts;
      }
      newMessages.push(m)
    }
    messages = newMessages;

    var newTools = [];
    if (tools) {
        for (var tool of tools) {
          const name = tool.function.name;
          const description = tool.function.description;
          const parameters = tool.function.parameters;
          newTools.push({
            name: name,
            description: description,
            input_schema: parameters
          })
        }
    }
    tools = newTools;
    console.log(tools);

    const claudeRequestBody = {
      messages: messages,
      system: systemPrompt,
      model: model,
      temperature: temperature,
      max_tokens: max_tokens,
      top_p: top_p,
      stop_sequences: stop,
      tools: tools
    };
    console.log(claudeRequestBody);

    const claudeResponse = await fetch(`${CLAUDE_BASE_URL}/v1/messages`, {
      method: 'POST',
      headers: {
        accept: 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(claudeRequestBody),
    });
    
    if (!claudeResponse.ok) {
      const errorText = await claudeResponse.text();
      console.error(`Error: ${errorText}`);
      throw new Error(`HTTP error! status: ${claudeResponse.status}, details: ${errorText}`);
    }

    const claudeResponseBody = await claudeResponse.json();
    console.log(claudeResponseBody);
    const openAIResponseBody = claudeToChatGPTResponse(claudeResponseBody);
    return new Response(JSON.stringify(openAIResponseBody), {
      status: claudeResponse.status,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

function handleOPTIONS() {
  return new Response(null, {
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': '*',
      'Access-Control-Allow-Headers': '*',
      'Access-Control-Allow-Credentials': 'true',
    },
  });
}
