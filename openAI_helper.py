import openai
def ask_computer(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100)
    return response['choices'][0]['text']
