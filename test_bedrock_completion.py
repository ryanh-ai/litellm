import pytest
import litellm
from litellm import completion, RateLimitError
from litellm.types.utils import ModelResponse
from tests.local_testing.test_bedrock_completion import messages, process_stream_response, encode_image


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
    ],
)
def test_completion_bedrock_nova_text_completion(model):
    print("calling bedrock nova completion - Text Only")
    try:
        response: ModelResponse = completion(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.1,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
    ],
)
def test_completion_bedrock_nova_text_completion_with_system(model):
    print("calling bedrock olympus completion - System with Text ")
    messages_with_system =[{"role": "system", "content": "You are a helpful assistant."}]
    messages_with_system.extend(messages)
    try:
        response: ModelResponse = completion(
            model=model,
            messages=messages_with_system,
            max_tokens=10,
            temperature=0.1,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0
        assert len(response.choices[0].message.content.split()) < 15


    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
    ],
)
def test_streaming_completion_bedrock_nova_text_completion_with_system(model):
    print("calling bedrock olympus completion - System with Text ")
    messages_with_system =[{"role": "system", "content": "You are a helpful assistant. You respond in french"}]
    messages_with_system.extend(messages)
    try:
        response: ModelResponse = completion(
            model=model,
            messages=messages_with_system,
            max_tokens=10,
            temperature=0.1,
            stream=True,
        )  # type: ignore
        response = process_stream_response(response, messages_with_system)
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0
        assert len(response.choices[0].message.content.split()) < 15

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/us.amazon.nova-pro-v1:0",
    ],
)
@pytest.mark.parametrize(
    "image",[
        ('jpeg', './road_in_jpeg.jpeg'),
        ('png', './sunset.png'),
    ]
)
def test_completion_nova_image_base64(model, image):
    try:
        litellm.num_retries = 3
        image_type, image_path = image
        # Getting the base64 string
        base64_image = encode_image(image_path)
        resp = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
    ],
)
def test_nova_tool_calling(model):
    """
    # related issue: https://github.com/BerriAI/litellm/issues/5007
    # Bedrock tool names must satisfy regular expression pattern: [a-zA-Z][a-zA-Z0-9_]* ensure this is true
    """
    litellm.set_verbose = True
    response = litellm.completion(
        model=model,
        temperature=1.0,
        top_p=1.0,
        top_k=1.0,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "-DoSomethingVeryCool-forLitellm_Testin999229291-0293993",
                    "description": "use this to get the current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    print("bedrock response")
    print(response)

    # Assert that the tools in response have the same function name as the input
    _choice_1 = response.choices[0]
    if _choice_1.message.tool_calls is not None:
        print(_choice_1.message.tool_calls)
        for tool_call in _choice_1.message.tool_calls:
            _tool_Call_name = tool_call.function.name
            if _tool_Call_name is not None and "DoSomethingVeryCool" in _tool_Call_name:
                assert (
                        _tool_Call_name
                        == "-DoSomethingVeryCool-forLitellm_Testin999229291-0293993"
                )


@pytest.mark.parametrize(
    "stop",
    [
        dict(prompt='count to ten with numerals within a markdown block, then explain what you did',
             stop="```"),
        dict(prompt="count to 10 with numerals, one line at a time starting with <tool> and when done end in </tool>, then explain what you did",
             stop="</tool>"),
        dict(prompt="count from 1 to 10 with numerals",
             stop="5"),
        dict(prompt="count from 1 to 10 with numerals, do not explain yourself just start counting",
             stop=["5", "10"])
     ],
)
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
    ],
)
def test_nova_stop_value(stop, model):
    try:
        litellm.set_verbose = False
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": stop['prompt']},
            ],
            "stop": stop['stop'],
        }
        response: ModelResponse = litellm.completion(
            model=model,
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].model_extra['finish_reason'] == 'stop'
        stop_seq = stop['stop'][0] if type(stop['stop']) == list else stop['stop']
        assert response.choices[0].message.content.endswith(stop_seq)

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
