const version = '0.4.0';

addEventListener('fetch', (event) => {
  event.respondWith(handleRequest(event.request));
});

const CLAUDE_API_KEY = ''; // Optional: default claude api key if you don't want to pass it in the request header
const CLAUDE_BASE_URL = 'https://api.anthropic.com'; // Change this if you are using a self-hosted endpoint
const MAX_TOKENS = 4096; // Max tokens to sample, change it if you want to sample more tokens, maximum is 4096.


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
        finish_reason: claudeResponse['stop_reason']
          ? stop_reason_map[claudeResponse['stop_reason']]
          : null,
      },
    ],
  };
  const message = {
    role: 'assistant',
    content: completion,
  };
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
    const { model, messages, temperature, stop } = requestBody;

    let systemPrompt = "The assistant is Claude created by Anthropic";
    let newMessages = messages;
    if (messages[0].role == "system") {
        systemPrompt = messages[0].content;
        newMessages = messages.slice(1);
    }

    const claudeRequestBody = {
      messages: newMessages,
      system: systemPrompt,
      model: model,
      temperature: temperature,
      max_tokens: MAX_TOKENS,
      stop_sequences: stop
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
