from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True,
)

memory.save_context(
    {"input": "こんにちは!"},
    {
        "output": "こんにちは!お元気ですか?何か質問があればどうぞお知らせください。どのようにお手伝いできますか?",
    },
)

memory.save_context(
    {
        "input": "今日はいい天気ですね",
    },
    {
        "output": "私はAIなので、実際の天候を感じることはできませんが、いい天気の日は外出や活動を楽しむのに最適ですね!",
    },
)

print(memory.load_memory_variables({}))
