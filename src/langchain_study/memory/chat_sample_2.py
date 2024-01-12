from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

result = chat.invoke(
    [
        HumanMessage(content="茶碗蒸しを作るのに必要な食材を教えて"),
        AIMessage(
            content="""茶碗蒸しを作るために必要な食材は以下の通りです:
1. 卵 - 茶碗蒸しの主成分であり、卵を使用して柔らかくふわふわとしたテクスチャーを作ります。
2. だし汁 - 茶碗蒸しのベースとなる出汁です。一般的には鰹節や昆布から作られた出汁が使用されますが、市販の出汁の素や顆粒出汁を使用することもできます。
3. 醤油 - 茶碗蒸しの味付けに使用される調味料です。日本の醤油が一般的ですが、お好みに合わせて濃いめの醤油や薄口醤油を使用することもできます。
4. 塩 - 醤油と一緒に茶碗蒸しの味付けに使用されます。お好みで調整してください。
5. 牛乳 - 茶碗蒸しの卵液を作るために使用されます。牛乳は卵液を滑らかにする役割を果たしますが、必須ではありません。
6. ごぼう - 茶碗蒸しに具として加えることがあります。ごぼうは細切りにして茹でたり炒めたりして使用しますが、必須ではありません。他の具材(例:鶏肉、えのき茸、シメジ、ほうれん草など)を使用することもできます。
7. その他の具材(任意) - 茶碗蒸しには他の具材を加えることもできます。例えば、鶏肉や海老、豆腐、野菜などを使うことがあります。

これらの食材を使用して、茶碗蒸しを作ることができます。具体的なレシピに従って材料を調理してください。
""",
        ),
        HumanMessage(content="前の回答を英語に翻訳して"),
    ],
)
print(result.content)