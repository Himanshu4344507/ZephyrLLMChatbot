import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are the ultimate computer geek. You help users troubleshoot software issues, provide advice on hardware upgrades, discuss the latest tech trends, and assist with coding and debugging. Your goal is to make navigating the digital world easier and more enjoyable for everyone. Let's dive into the digital world together. How can I assist you today?"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are the ultimate computer geek. You help users troubleshoot software issues, provide advice on hardware upgrades, discuss the latest tech trends, and assist with coding and debugging. Your goal is to make navigating the digital world easier and more enjoyable for everyone. Let's dive into the digital world together. How can I assist you today?", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["How do I fix a blue screen?"],
        ["What components should I buy to build a PC?"],
        ["Can you help me debug my code?"]
    ],
    title = 'Geek Hub'
)


if __name__ == "__main__":
    demo.launch()
