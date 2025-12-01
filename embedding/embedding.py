import os
import glob
import json
import time
import google.generativeai as genai
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 환경 변수 로드 및 Google API 설정
load_dotenv()
GOOGLE_API_KEY = ""

genai.configure(api_key=GOOGLE_API_KEY)

# 모델 설정 (만약 3.0이 안 되면 'gemini-1.5-pro' 또는 'gemini-1.5-flash'로 변경하세요)
MODEL_NAME = 'gemini-2.5-flash' 
print(f"사용 중인 모델: {MODEL_NAME}")
model = genai.GenerativeModel(MODEL_NAME)

def parse_pdf_with_gemini(file_path):
    """
    PDF를 페이지별 이미지로 변환 후 Gemini에게 마크다운 변환을 요청
    """
    print(f"1. PDF를 이미지로 변환 중... ({os.path.basename(file_path)})")
    
    try:
        # dpi=300이 너무 느리면 150으로 낮춰도 텍스트 인식에는 충분할 수 있습니다.
        pages = convert_from_path(file_path, dpi=300)
    except Exception as e:
        print(f"Error converting PDF to image: {e}")
        return ""

    full_markdown_text = ""
    
    print(f"2. Gemini를 사용하여 {len(pages)} 페이지 파싱 시작...")
    
    prompt = """
    당신은 완벽한 문서 파서입니다. 아래 제공된 이미지(매뉴얼 페이지)를 분석하여 텍스트로 변환하세요.
    
    [지시사항]
    1. 모든 텍스트를 **한국어 Markdown 형식**으로 출력하세요.
    2. **표(Table)**는 반드시 Markdown Table 문법(| header | ... |)을 사용하여 구조를 완벽히 유지하세요.
    3. **제목과 소제목**은 문서의 위계에 맞춰 #, ##, ### 헤더를 정확히 사용하세요.
    4. '경고', '주의' 같은 박스 내용은 > (Blockquote)를 사용하여 강조하세요.
    5. 이미지나 아이콘에 대한 불필요한 묘사는 생략하고 텍스트 내용에 집중하세요.
    6. 페이지 번호나 머리말/꼬리말은 제외하세요.
    """

    for i, page_image in enumerate(pages):
        print(f"   - Converting Page {i + 1}/{len(pages)}...", end=" ")
        
        try:
            # Gemini 호출
            response = model.generate_content([prompt, page_image])
            full_markdown_text += response.text + "\n\n"
            print("Done.")
            
            # API Rate Limit 방지를 위한 짧은 대기 (필요 시 조절)
            time.sleep(2) 
            
        except Exception as e:
            print(f"\n   ! Error on page {i+1}: {e}")
            continue

    return full_markdown_text

def process_laundry_manual_google(file_path, device_type):
    """
    파싱 및 청킹 수행
    """
    # 1. 파싱
    markdown_text = parse_pdf_with_gemini(file_path)
    
    if not markdown_text:
        return []

    # 2. 1차 청킹 (헤더 기준)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )
    
    header_splits = markdown_splitter.split_text(markdown_text)
    
    # 3. 2차 청킹 (길이 기준)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = text_splitter.split_documents(header_splits)
    
    # 4. 메타데이터 태깅
    processed_docs = []
    for doc in final_chunks:
        doc.metadata["device_type"] = device_type
        doc.metadata["source_file"] = os.path.basename(file_path)
        
        if "|" in doc.page_content and "---" in doc.page_content:
            doc.metadata["contains_table"] = True
            
        processed_docs.append(doc)
        
    return processed_docs

def save_results_to_file(documents, output_dir="parsed_results"):
    """
    결과를 JSON(데이터용)과 Markdown(확인용) 파일로 저장
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. JSON으로 저장 (나중에 DB 로드용으로 좋음)
    json_output_path = os.path.join(output_dir, "all_parsed_data.json")
    
    # Document 객체를 dict로 변환
    docs_dict = [
        {"page_content": doc.page_content, "metadata": doc.metadata} 
        for doc in documents
    ]
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(docs_dict, f, ensure_ascii=False, indent=4)
    print(f"\n✅ JSON 데이터 저장 완료: {json_output_path}")

    # 2. Markdown으로 저장 (사람이 눈으로 확인하기 좋음)
    md_output_path = os.path.join(output_dir, "readable_check.md")
    
    with open(md_output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Metadata: {doc.metadata}\n\n")
            f.write(doc.page_content)
            f.write("\n\n")
    print(f"✅ Markdown 확인 파일 저장 완료: {md_output_path}")

# --- 실행부 ---

if __name__ == "__main__":
    # 현재 파일 위치 기준 assets 폴더 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    
    # assets 폴더가 없으면 생성 안내
    if not os.path.exists(assets_dir):
        print(f"⚠️ 'assets' 폴더가 없습니다. {assets_dir} 경로에 폴더를 만들고 PDF를 넣어주세요.")
        exit()

    files = glob.glob(os.path.join(assets_dir, "*.pdf"))
    
    if not files:
        print("⚠️ 'assets' 폴더에 PDF 파일이 없습니다.")
        exit()

    all_documents = []

    # 모든 PDF 파일 처리
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"\n=== [{file_name}] 처리 시작 ===")
        
        device_type = os.path.splitext(file_name)[0]
        
        file_docs = process_laundry_manual_google(file_path, device_type)
        all_documents.extend(file_docs)
        print(f"=== [{file_name}] 처리 완료: {len(file_docs)} chunks 생성 ===")

    # 결과 저장 및 확인
    if all_documents:
        save_results_to_file(all_documents)
        
        print("\n[샘플 데이터 확인 (상위 1개)]")
        print("-" * 50)
        print(f"Metadata: {all_documents[0].metadata}")
        print(f"Content Preview:\n{all_documents[0].page_content[:300]}...")
        print("-" * 50)
    else:
        print("\n❌ 처리된 데이터가 없습니다.")