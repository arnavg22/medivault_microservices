"""
Advanced Semantic Chunking for Medical Documents
Production-grade: Section-aware, sentence-boundary chunking with overlap
"""
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalTextSplitter:
    """
    Intelligent text splitter for medical documents.
    
    Strategy:
    1. Identify semantic sections (medications, diagnosis, labs, etc.)
    2. Chunk by section, not by arbitrary size
    3. Add 10-20% overlap between chunks for context continuity
    4. Attach rich metadata to each chunk
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap_percent: float = None,
        use_sentence_boundaries: bool = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        overlap_pct = chunk_overlap_percent or settings.chunk_overlap_percent
        self.chunk_overlap = int(self.chunk_size * overlap_pct)
        self.use_sentence_boundaries = use_sentence_boundaries or settings.use_sentence_boundaries
        
        logger.info(
            f"Initialized MedicalTextSplitter (Production): "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap} ({int(overlap_pct*100)}%), "
            f"sentence_aware={self.use_sentence_boundaries}"
        )
        
        # Medical section patterns - comprehensive
        self.section_patterns = {
            'patient_info': r'(?i)(patient\s+(?:information|details|demographics)|name|age|gender|dob)',
            'chief_complaint': r'(?i)(chief\s+complaint|presenting\s+complaint|reason\s+for\s+visit)',
            'medications': r'(?i)(medications?|prescriptions?|drugs?|current\s+medications)[\s:]*',
            'diagnosis': r'(?i)(diagnosis|diagnoses|impression|assessment)[\s:]*',
            'symptoms': r'(?i)(symptoms?|complaints?|presenting\s+symptoms?)[\s:]*',
            'vitals': r'(?i)(vital\s+signs?|vitals?)[\s:]*',
            'lab_results': r'(?i)(lab(?:oratory)?\s+results?|test\s+results?|investigations?)[\s:]*',
            'medical_history': r'(?i)(medical\s+history|patient\s+history|past\s+medical\s+history)[\s:]*',
            'allergies': r'(?i)(allergies|allergic\s+to|known\s+allergies?)[\s:]*',
            'procedures': r'(?i)(procedures?|operations?|surgeries|surgical\s+history)[\s:]*',
            'doctor_notes': r'(?i)(doctor[\'s]?\s+notes?|physician\s+notes?|clinical\s+notes?)[\s:]*',
            'follow_up': r'(?i)(follow[\-\s]?up|recommendations?|next\s+steps?|plan)[\s:]*',
        }
        
        # Base splitter for general content
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    
    def split_document(self, document: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Split a medical document into semantic chunks with metadata.
        
        Production features:
        - Table-aware: Tables kept as whole chunks
        - Page tracking: Extract page numbers from text
        - Rich metadata: doctor, hospital, report type, etc.
        
        Args:
            document: Dict with 'text', 'source', 'filename', optional metadata
            
        Returns:
            List of chunk dictionaries with text and rich metadata
        """
        text = document.get("text", "")
        
        if not text.strip():
            logger.warning("Empty document received")
            return []
        
        # Step 0: Extract tables first (production requirement: keep tables whole)
        table_chunks = self._extract_table_chunks(text, document)
        
        # Remove table text from main text
        text_without_tables = self._remove_tables_from_text(text)
        
        # Step 1: Identify sections in remaining text
        sections = self._identify_sections(text_without_tables)
        
        # Step 2: Create chunks from sections with overlap
        text_chunks = []
        chunk_id = 0
        
        for section_type, section_text, section_start, section_end in sections:
            section_chunks = self._chunk_section_with_overlap(
                section_text,
                section_type
            )
            
            for chunk_text in section_chunks:
                # Extract page number from chunk text
                page_num = self._extract_page_number(chunk_text)
                
                text_chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "chunk_type": "text",
                    "section_type": section_type,
                    "page": page_num,
                    "source": document.get("source", ""),
                    "filename": document.get("filename", ""),
                    "date": document.get("date", None),
                    "extraction_method": document.get("extraction_method", "unknown"),
                    "position_in_doc": section_start,
                    # Rich metadata from document
                    "doctor_name": document.get("doctor_name"),
                    "hospital_name": document.get("hospital_name"),
                    "report_date": document.get("report_date"),
                    "report_type": document.get("report_type"),
                    "patient_id": document.get("patient_id"),
                })
                chunk_id += 1
        
        # Step 3: Process any remaining text not in identified sections
        remaining_text = self._get_remaining_text(text_without_tables, sections)
        if remaining_text.strip():
            general_chunks = self._chunk_with_overlap(
                remaining_text,
                "general"
            )
            
            for chunk_text in general_chunks:
                page_num = self._extract_page_number(chunk_text)
                
                text_chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "chunk_type": "text",
                    "section_type": "general",
                    "page": page_num,
                    "source": document.get("source", ""),
                    "filename": document.get("filename", ""),
                    "date": document.get("date", None),
                    "extraction_method": document.get("extraction_method", "unknown"),
                    "position_in_doc": 0,
                    # Rich metadata
                    "doctor_name": document.get("doctor_name"),
                    "hospital_name": document.get("hospital_name"),
                    "report_date": document.get("report_date"),
                    "report_type": document.get("report_type"),
                    "patient_id": document.get("patient_id"),
                })
                chunk_id += 1
        
        # Combine table chunks and text chunks
        all_chunks = table_chunks + text_chunks
        
        logger.info(
            f"Created {len(all_chunks)} chunks ({len(table_chunks)} tables, "
            f"{len(text_chunks)} text) from '{document.get('filename', 'unknown')}'"
        )
        return all_chunks
        
        logger.info(f"Created {len(chunks)} chunks from document '{document.get('filename', 'unknown')}'")
        return chunks
    
    def _extract_table_chunks(self, text: str, document: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Extract table chunks from text. Tables are kept WHOLE (never split).
        
        Production requirement: Lab results and vital tables must stay intact.
        
        Args:
            text: Full document text
            document: Document metadata
            
        Returns:
            List of table chunks
        """
        table_chunks = []
        
        # Pattern to find tables: [Table X on Page Y]
        table_pattern = r'\[Table (\d+) on Page (\d+)\]([^\[]+)'
        
        for match in re.finditer(table_pattern, text, re.DOTALL):
            table_num = match.group(1)
            page_num = int(match.group(2))
            table_text = match.group(3).strip()
            
            # Only include substantial tables
            if len(table_text) > 30:
                table_chunks.append({
                    "text": match.group(0),  # Include header for context
                    "chunk_id": -1,  # Will be reassigned later
                    "chunk_type": "table",
                    "section_type": "table",
                    "page": page_num,
                    "table_number": int(table_num),
                    "source": document.get("source", ""),
                    "filename": document.get("filename", ""),
                    "date": document.get("date", None),
                    "extraction_method": document.get("extraction_method", "unknown"),
                    "position_in_doc": match.start(),
                    # Rich metadata
                    "doctor_name": document.get("doctor_name"),
                    "hospital_name": document.get("hospital_name"),
                    "report_date": document.get("report_date"),
                    "report_type": document.get("report_type"),
                    "patient_id": document.get("patient_id"),
                })
        
        logger.info(f"Extracted {len(table_chunks)} table chunks (kept whole)")
        return table_chunks
    
    def _remove_tables_from_text(self, text: str) -> str:
        """
        Remove table sections from text to avoid duplicate chunking.
        
        Args:
            text: Full document text
            
        Returns:
            Text with tables removed
        """
        # Remove [Table X on Page Y] sections
        cleaned_text = re.sub(
            r'\[Table \d+ on Page \d+\][^\[]+',
            '',
            text,
            flags=re.DOTALL
        )
        return cleaned_text
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """
        Extract page number from chunk text.
        
        Looks for [Page X] markers inserted during PDF extraction.
        
        Args:
            text: Chunk text
            
        Returns:
            Page number or None
        """
        match = re.search(r'\[Page (\d+)\]', text)
        if match:
            return int(match.group(1))
        return None
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Identify semantic sections in the document.
        
        Returns:
            List of (section_type, section_text, start_pos, end_pos) tuples
        """
        sections = []
        
        for section_type, pattern in self.section_patterns.items():
            for match in re.finditer(pattern, text):
                start_idx = match.start()
                
                # Find the end of this section
                end_idx = self._find_section_end(text, start_idx, pattern)
                
                section_text = text[start_idx:end_idx].strip()
                
                # Only include sections with substantial content
                if len(section_text) > 30:
                    sections.append((section_type, section_text, start_idx, end_idx))
        
        # Remove overlapping sections (keep the first identified)
        sections = self._remove_overlapping_sections(sections)
        
        # Sort by position in document
        sections.sort(key=lambda x: x[2])
        
        return sections
    
    def _find_section_end(self, text: str, start_idx: int, current_pattern: str) -> int:
        """
        Find where a section ends by looking for the next section or significant gap.
        """
        remaining = text[start_idx:]
        
        # Start with the full remaining text
        end_pos = len(remaining)
        
        # Look for the next section header (skip first 50 chars to avoid matching current header)
        for section_type, pattern in self.section_patterns.items():
            if pattern == current_pattern:
                continue  # Skip current section type
            
            match = re.search(pattern, remaining[50:])
            if match:
                end_pos = min(end_pos, match.start() + 50)
        
        # Also look for multiple blank lines (common section separator)
        blank_lines_match = re.search(r'\n\s*\n\s*\n', remaining)
        if blank_lines_match:
            end_pos = min(end_pos, blank_lines_match.start())
        
        # Don't make sections too long (max 2x chunk_size)
        max_section_length = self.chunk_size * 2
        end_pos = min(end_pos, max_section_length)
        
        return start_idx + end_pos
    
    def _remove_overlapping_sections(
        self, 
        sections: List[Tuple[str, str, int, int]]
    ) -> List[Tuple[str, str, int, int]]:
        """Remove overlapping sections, keeping the first identified."""
        if not sections:
            return []
        
        non_overlapping = []
        
        for section in sections:
            section_type, section_text, start, end = section
            
            # Check if this section overlaps with any already added
            overlaps = False
            for existing in non_overlapping:
                _, _, ex_start, ex_end = existing
                
                # Check for overlap
                if not (end <= ex_start or start >= ex_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(section)
        
        return non_overlapping
    
    def _chunk_section_with_overlap(
        self, 
        section_text: str, 
        section_type: str
    ) -> List[str]:
        """
        Chunk a section with appropriate overlap strategy.
        
        For critical sections (medications, diagnosis), use special handling.
        """
        # If section fits in one chunk, return as-is
        if len(section_text) <= self.chunk_size:
            return [section_text]
        
        # Special handling for list-based sections
        if section_type in ['medications', 'allergies', 'procedures']:
            return self._chunk_list_with_overlap(section_text)
        
        # For other sections, use overlapping text splitter
        return self._chunk_with_overlap(section_text, section_type)
    
    def _chunk_list_with_overlap(self, section_text: str) -> List[str]:
        """
        Chunk list-based sections (medications, etc.) ensuring complete items
        stay together and chunks have overlap.
        
        This ensures: "I only get 5 medicines instead of 10" problem is solved.
        """
        # Detect list items (numbered, bulleted, or capitalized lines)
        items = re.split(
            r'\n(?=\d+[\.\)]\s|\-\s|\*\s|[A-Z][a-z]+:)',
            section_text
        )
        
        if len(items) <= 1:
            # Not a list, use regular chunking
            return self._chunk_with_overlap(section_text, "list")
        
        chunks = []
        current_chunk_items = []
        current_length = 0
        
        # Extract header (first element if it's not a list item)
        header = ""
        if items and not re.match(r'^\d+[\.\)]|\-|\*', items[0].strip()):
            header = items[0].strip() + "\n\n"
            items = items[1:]
        
        for i, item in enumerate(items):
            item_length = len(item)
            
            # If adding this item would exceed chunk size, create a chunk
            if current_length + item_length > self.chunk_size and current_chunk_items:
                # Create chunk with current items
                chunk_text = header + "\n".join(current_chunk_items)
                chunks.append(chunk_text)
                
                # Calculate overlap: keep last ~15% of items for context
                overlap_count = max(1, len(current_chunk_items) // 7)  # ~15%
                overlap_items = current_chunk_items[-overlap_count:]
                
                # Start new chunk with overlap
                current_chunk_items = overlap_items
                current_length = sum(len(item) for item in overlap_items)
            
            current_chunk_items.append(item)
            current_length += item_length
        
        # Add remaining items as final chunk
        if current_chunk_items:
            chunk_text = header + "\n".join(current_chunk_items)
            chunks.append(chunk_text)
        
        return chunks if chunks else [section_text]
    
    def _chunk_with_overlap(self, text: str, section_type: str) -> List[str]:
        """
        Chunk text with overlap using sentence-boundary aware splitting (production).
        
        Args:
            text: Text to chunk
            section_type: Type of section
            
        Returns:
            List of text chunks with proper sentence boundaries
        """
        if self.use_sentence_boundaries:
            return self._sentence_aware_chunk(text)
        else:
            # Fallback to basic recursive splitting
            return self.base_splitter.split_text(text)
    
    def _sentence_aware_chunk(self, text: str) -> List[str]:
        """
        Production-grade sentence-boundary aware chunking.
        
        Ensures:
        - Chunks don't split mid-sentence
        - Overlap maintains context
        - Medical abbreviations handled correctly (e.g., "Dr.", "mg.", "B.P.")
        """
        # Split into sentences (medical-aware)
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap: include last few sentences for context
                overlap_sentences = self._get_overlap_sentences(current_chunk, self.chunk_overlap)
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks if chunks else [text]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with medical abbreviation handling.
        
        Handles:
        - Standard sentence boundaries (. ! ?)
        - Medical abbreviations (Dr., mg., ml., etc.)
        - Decimal numbers (12.5, 3.14)
        """
        # Medical abbreviations that shouldn't trigger sentence splits
        medical_abbrevs = r'(?:Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|e\.g|i\.e|mg|ml|cm|mm|kg|lb|oz|b\.p|temp)'
        
        # Replace medical abbreviations temporarily
        protected_text = re.sub(
            rf'\b({medical_abbrevs})\.',
            lambda m: m.group(1).replace('.', '<DOT>'),
            text,
            flags=re.IGNORECASE
        )
        
        # Split on sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, protected_text)
        
        # Restore protected dots
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], target_overlap_chars: int) -> List[str]:
        """
        Get last N sentences that fit within target overlap character count.
        
        Args:
            sentences: List of sentences
            target_overlap_chars: Target overlap in characters
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
        
        overlap = []
        char_count = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            if char_count + len(sentence) > target_overlap_chars:
                break
            overlap.insert(0, sentence)
            char_count += len(sentence)
        
        # Ensure at least one sentence for context
        if not overlap and sentences:
            overlap = [sentences[-1]]
        
        return overlap
    
    def _create_adaptive_splitter(self, text: str, section_type: str) -> List[str]:
        """
        Create adaptive text splitter with section-specific overlap.
        Ensures ~10-20% overlap for context continuity.
        
        Args:
            text: Text to split
            section_type: Type of section (affects overlap strategy)
            
        Returns:
            List of text chunks
        """
        # Use larger overlap for critical sections
        overlap = self.chunk_overlap
        if section_type in ['diagnosis', 'doctor_notes', 'follow_up']:
            overlap = int(self.chunk_size * 0.20)  # 20% overlap for critical sections
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        
        return splitter.split_text(text)
    
    def _get_remaining_text(
        self, 
        full_text: str, 
        sections: List[Tuple[str, str, int, int]]
    ) -> str:
        """
        Extract text that wasn't part of identified sections.
        """
        if not sections:
            return full_text
        
        # Sort sections by position
        sections_sorted = sorted(sections, key=lambda x: x[2])
        
        remaining_parts = []
        last_end = 0
        
        for _, _, start, end in sections_sorted:
            # Add text between last section and this one
            if start > last_end:
                remaining_parts.append(full_text[last_end:start])
            last_end = end
        
        # Add any text after the last section
        if last_end < len(full_text):
            remaining_parts.append(full_text[last_end:])
        
        return "\n\n".join(part.strip() for part in remaining_parts if part.strip())
    
    def batch_split(self, documents: List[Dict]) -> List[Dict]:
        """
        Split multiple documents.
        Fully dynamic - handles any number of documents.
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.split_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to split document {doc.get('filename', 'unknown')}: {e}")
        
        logger.info(f"Batch split complete: {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks


# Example usage
if __name__ == "__main__":
    splitter = MedicalTextSplitter(chunk_size=600, chunk_overlap_percent=0.15)
    
    sample_doc = {
        "text": """
Patient Information:
Name: John Doe
Age: 45 years
Gender: Male

Chief Complaint:
Patient presents with chest pain and shortness of breath.

Medications:
1. Metformin 500mg - twice daily with meals
2. Lisinopril 10mg - once daily in the morning
3. Aspirin 81mg - once daily
4. Atorvastatin 20mg - once daily at bedtime
5. Omeprazole 20mg - once daily before breakfast
6. Levothyroxine 50mcg - once daily on empty stomach
7. Metoprolol 25mg - twice daily
8. Gabapentin 300mg - three times daily

Diagnosis:
1. Type 2 Diabetes Mellitus - well controlled
2. Hypertension - Stage 1
3. Hyperlipidemia
4. Hypothyroidism

Lab Results:
HbA1c: 6.8%
Fasting Glucose: 125 mg/dL
Total Cholesterol: 180 mg/dL
LDL: 100 mg/dL
HDL: 55 mg/dL
TSH: 2.5 mIU/L

Follow-up:
Return in 3 months for medication review and lab work.
Continue current medications.
Lifestyle modifications recommended.
        """,
        "source": "sample.pdf",
        "filename": "sample.pdf",
        "date": "2024-01-15"
    }
    
    chunks = splitter.split_document(sample_doc)
    print(f"\n{'='*60}")
    print(f"Generated {len(chunks)} chunks:")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({chunk['section_type']}) ---")
        print(f"Length: {len(chunk['text'])} characters")
        print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
