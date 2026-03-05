"""
Create a realistic test PDF for MediVault RAG Bot testing
Generates a medical prescription document with structured content
"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from pathlib import Path
import os


def create_test_prescription():
    """Create a realistic medical prescription PDF"""
    
    # Ensure output directory exists
    output_dir = Path("data/raw_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "test_prescription.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=50,
        bottomMargin=50
    )
    
    # Container for the document
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a472a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    header_style = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#2c5f2d'),
        alignment=TA_CENTER
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1a472a'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    
    # Hospital Header
    story.append(Paragraph("MediVault General Hospital", title_style))
    story.append(Paragraph("123 Healthcare Avenue, Medical District", header_style))
    story.append(Paragraph("Phone: (555) 123-4567 | Fax: (555) 123-4568", header_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Prescription Title
    story.append(Paragraph("MEDICAL PRESCRIPTION", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Patient Information
    story.append(Paragraph("PATIENT INFORMATION", section_title_style))
    patient_data = [
        ["Patient Name:", "John Doe", "Date:", "March 20, 2025"],
        ["Patient ID:", "MV-2025-00123", "Age:", "45 years"],
        ["Gender:", "Male", "Blood Type:", "O+"]
    ]
    patient_table = Table(patient_data, colWidths=[1.5*inch, 2*inch, 1*inch, 1.5*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a472a')),
        ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#1a472a')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.2 * inch))
    
    # Diagnosis
    story.append(Paragraph("DIAGNOSIS", section_title_style))
    story.append(Paragraph("• Primary: <b>Hypertension</b> (ICD-10: I10)", normal_style))
    story.append(Paragraph("• Secondary: <b>Hyperlipidemia</b> (ICD-10: E78.5)", normal_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Vital Signs
    story.append(Paragraph("VITAL SIGNS", section_title_style))
    vitals_data = [
        ["Blood Pressure:", "145/92 mmHg", "(HIGH)"],
        ["Heart Rate:", "78 bpm", "(Normal)"],
        ["Temperature:", "98.4°F", "(Normal)"],
        ["Weight:", "185 lbs", ""],
        ["BMI:", "27.8", "(Overweight)"]
    ]
    vitals_table = Table(vitals_data, colWidths=[1.8*inch, 1.5*inch, 1.5*inch])
    vitals_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (2, 0), (2, 0), colors.red),  # HIGH in red
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(vitals_table)
    story.append(Spacer(1, 0.2 * inch))
    
    # Medications
    story.append(Paragraph("MEDICATIONS PRESCRIBED", section_title_style))
    med_data = [
        ["Medication", "Dosage", "Frequency", "Duration"],
        ["Amlodipine", "5mg", "Once daily (morning)", "30 days"],
        ["Metoprolol", "25mg", "Twice daily (with meals)", "30 days"],
        ["Atorvastatin", "20mg", "Once daily (evening)", "30 days"],
    ]
    med_table = Table(med_data, colWidths=[1.8*inch, 1*inch, 1.8*inch, 1*inch])
    med_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8f5e9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a472a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(med_table)
    story.append(Spacer(1, 0.2 * inch))
    
    # Lab Results
    story.append(Paragraph("LABORATORY RESULTS", section_title_style))
    lab_data = [
        ["Test", "Result", "Reference Range", "Status"],
        ["Total Cholesterol", "210 mg/dL", "< 200 mg/dL", "HIGH"],
        ["LDL Cholesterol", "142 mg/dL", "< 100 mg/dL", "HIGH"],
        ["HDL Cholesterol", "48 mg/dL", "> 40 mg/dL", "Normal"],
        ["Triglycerides", "158 mg/dL", "< 150 mg/dL", "Borderline"],
        ["Fasting Glucose", "105 mg/dL", "70-100 mg/dL", "Borderline"],
    ]
    lab_table = Table(lab_data, colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 1.1*inch])
    lab_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fff3e0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#e65100')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TEXTCOLOR', (3, 1), (3, 2), colors.red),  # HIGH status in red
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(lab_table)
    story.append(Spacer(1, 0.2 * inch))
    
    # Allergies
    story.append(Paragraph("ALLERGIES", section_title_style))
    story.append(Paragraph("• <b>Penicillin</b> - Severe reaction (rash, difficulty breathing)", normal_style))
    story.append(Paragraph("• <b>Sulfa drugs</b> - Moderate reaction (hives)", normal_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Instructions
    story.append(Paragraph("PATIENT INSTRUCTIONS", section_title_style))
    instructions = [
        "Take Amlodipine in the morning with or without food",
        "Take Metoprolol with meals to reduce stomach upset",
        "Take Atorvastatin in the evening for maximum effectiveness",
        "Monitor blood pressure daily at home and record readings",
        "Follow low-sodium diet (< 2000mg sodium per day)",
        "Exercise for 30 minutes, 5 days per week (walking, swimming)",
        "Reduce alcohol consumption and avoid smoking",
        "Schedule lab work in 3 months to recheck cholesterol levels"
    ]
    for instruction in instructions:
        story.append(Paragraph(f"• {instruction}", normal_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # Follow-up
    story.append(Paragraph("FOLLOW-UP APPOINTMENT", section_title_style))
    story.append(Paragraph("<b>Date:</b> April 3, 2025 (2 weeks)", normal_style))
    story.append(Paragraph("<b>Purpose:</b> Check blood pressure control and medication tolerance", normal_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Doctor signature
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("_" * 40, normal_style))
    story.append(Paragraph("<b>Dr. Arjun Mehta, MD</b>", normal_style))
    story.append(Paragraph("Cardiology Department", normal_style))
    story.append(Paragraph("License #: MD-12345", normal_style))
    story.append(Paragraph("Date: March 20, 2025", normal_style))
    
    # Build PDF
    doc.build(story)
    
    print(f"✅ Test PDF created: {output_path}")
    print(f"   File size: {os.path.getsize(output_path)} bytes")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Test Medical PDF")
    print("=" * 60)
    
    try:
        pdf_path = create_test_prescription()
        print("\n🎉 Test PDF ready for use in test_routes.py")
        print(f"\nThe PDF contains:")
        print("  • Patient information")
        print("  • Diagnosis: Hypertension, Hyperlipidemia")
        print("  • Medications: Amlodipine, Metoprolol, Atorvastatin")
        print("  • Lab results with reference ranges")
        print("  • Vital signs table")
        print("  • Allergies information")
        print("  • Follow-up instructions")
    except Exception as e:
        print(f"\n❌ Error creating PDF: {e}")
        print("\nMake sure reportlab is installed:")
        print("  pip install reportlab")
