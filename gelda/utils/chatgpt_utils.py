import re
import openai
from wrapt_timeout_decorator import *

CHATGPT_TIMEOUT = 100  # seconds
DEFAULT_CHATGPT_KWARGS = {"model": "gpt-3.5-turbo",
                          "temperature": 0.3,
                          "max_tokens": 1024}


@timeout(CHATGPT_TIMEOUT)
def _get_chatgpt_response(**chat_kwargs):
    """
    Get response from chatgpt. Timeout after CHATGPT_TIMEOUT seconds.
    """
    response = openai.ChatCompletion.create(**chat_kwargs)
    return response


def get_chatgpt_response(n_attempts=0, **chat_kwargs):
    if n_attempts == 0:
        response = _get_chatgpt_response(**chat_kwargs)
    else:
        attempt_count = 0
        while attempt_count < n_attempts:
            try:
                response = _get_chatgpt_response(**chat_kwargs)
                if response is not None:
                    break
                else:
                    print(
                        f"(attempt {attempt_count + 1}/{n_attempts}) no chatgpt response... trying again...")
                    attempt_count += 1
                    continue
            except Exception:
                print(
                    f"(attempt {attempt_count + 1}/{n_attempts}) no chatgpt response after {CHATGPT_TIMEOUT} seconds... trying again...")
                attempt_count += 1
                continue
    return response

def get_list_from_chatgpt(query_messages: list, n_attempts=10, **chat_kwargs):
    """
    Get list from chatgpt. Timeout after CHATGPT_TIMEOUT seconds.
    """
    # initialize outputs
    response_messages = []
    list_items = None

    # run chat gpt completion for n attempts
    for i in range(n_attempts):
        # print(f"attempt {count + 1} to generate list from chatgpt...")

        try:
            # request chat completion.
            response = get_chatgpt_response(messages=query_messages, **chat_kwargs)
            # time.sleep(1)  # sleep for 1 second to avoid rate limit?
            message = response['choices'][-1]['message']["content"]
            response_messages.append(message)

            # try to parse message
            if "```python" in message:
                python_message = message.split("```python")[-1].replace("```", "").replace('\n', "").replace("_", " ")
                list_items = python_message[python_message.index("[") + 1:python_message.index("]")].split(',')
                list_items = [item.strip().strip('\""').strip('\'').lower() for item in list_items]
                break
            elif "Python list: " in message:
                python_message = message.split("Python list: ")[-1].replace('\n', "").replace("_", " ")
                list_items = python_message[python_message.index("[") + 1:python_message.index("]")].split(',')
                list_items = [item.strip().strip('\""').strip('\'').lower() for item in list_items]
                break
            elif "Python list representation:" in message:
                python_message = message.split("Python list representation:")[-1].replace('\n', "").replace("_", " ")
                list_items = python_message[python_message.index("[") + 1:python_message.index("]")].split(',')
                list_items = [item.strip().strip('\""').strip('\'').lower() for item in list_items]
                break
            elif message[0] == "[" and message[-1] == "]":
                list_items = message[1:-1].split(',')
                list_items = [item.strip().strip('\""').strip('\'').lower() for item in list_items]
                break
            elif "[" in message and "]" in message:
                list_items = message[message.index("[") + 1:message.index("]")].split(',')
                list_items = [item.strip().strip('\""').strip('\'').lower() for item in list_items]
                break
            elif len(re.findall('[\d]', message)) > 0:
                list_items = re.split("[\d]", message)[1::]
                list_items = [item.replace('.', '').replace('\"', '').replace('\'', '').strip().lower()
                              for item in list_items]
                break
            else:
                print(f"(attempt {i + 1}/{n_attempts}) could not parse chatgpt response... trying again...")

        except Exception:
            print(f"(attempt {i + 1}/{n_attempts}) chatgpt response took longer than {CHATGPT_TIMEOUT} seconds... trying again...")
            continue

    # remove some punctuation from list items ('?!.,;:')
    if list_items is not None:
        list_items = [item.translate(str.maketrans('', '', '?!.,;'))
                      for item in list_items]

    return list_items, response_messages
