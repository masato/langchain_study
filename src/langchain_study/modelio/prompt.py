from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか?",
    input_variables=[
        "product",
    ],
)

print(prompt.format())
print(prompt.format(product="Xperia"))
