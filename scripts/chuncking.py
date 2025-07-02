from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_clean_book(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Optional: remove front matter like TOC or publisher pages
    toc_index = raw_text.find("on the final day of my sophomore year")
    if toc_index != -1:
        raw_text = raw_text[toc_index:]  # start from real content

    return raw_text

def chunk_text(text, chunk_size=1500, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def save_chunks(chunks, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"[Chunk {i+1}]\n{chunk}\n\n")

if __name__ == "__main__":
    input_file = os.path.join("..", "data", "cleaned", "AtomicHabits.txt")
    output_file = os.path.join("..", "data", "cleaned", "chunks.txt")

    text = load_and_clean_book(input_file)
    chunks = chunk_text(text)
    save_chunks(chunks, output_file)

    print(f"[✓] Created {len(chunks)} chunks at: {output_file}")
 