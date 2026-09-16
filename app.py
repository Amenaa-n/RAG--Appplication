import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# جلب مفتاح API من متغيرات البيئة (الأسلوب الموصى به لـ GitHub لتجنب تسريب المفاتيح)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. نص النتائج الطبية (مصدر البيانات)
medical_reports_data = """
تقرير تحاليل طبية - المريض: أحمد محمود
التاريخ: 2026-09-10

1. تحليل صورة الدم الكاملة (CBC):
- الهيموجلوبين (Hemoglobin): 11.2 g/dL (المنسوب منخفض، الطبيعي 13.5-17.5).
- كريات الدم البيضاء (WBC): 6,500 /mcL (طبيعي).
- الصفائح الدموية (Platelets): 250,000 /mcL (طبيعي).

2. تحليل فحص السكر:
- السكر الصائم (Fasting Blood Sugar): 105 mg/dL (مرحلة ما قبل السكري).
- التراكمي (HbA1c): 5.8% (طبيعي إلى مرتفع قليلاً).

3. وظائف الكلى:
- الكرياتينين (Creatinine): 0.9 mg/dL (طبيعي).
- اليوريا (Urea): 28 mg/dL (طبيعي).

التوصية العامة: يُنصح بزيادة الأطعمة الغنية بالحديد ومتابعة نسبة السكر مع الطبيب المختص.
"""

# 2. تقسيم النص والـ Embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.create_documents([medical_reports_data])

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. إعداد النموذج والـ Prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

system_prompt = (
"أنت مساعد طبي ذكي. قم بالإجابة على أسئلة المستخدم بناءً على التقرير الطبي المرفق فقط.\n"
"إذا لم تجد الإجابة في التقرير، قل بوضوح أن المعلومات غير متوفرة.\n\n"
"التقرير الطبي المرفق:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{input}"),
])

# 4. بناء الـ RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if __name__ == "__main__":
query = "ما هي حالة الهيموجلوبين والسكر لدى المريض؟"
response = rag_chain.invoke({"input": query})
print("السؤال:", query)
print("الإجابة:", response["answer"])
