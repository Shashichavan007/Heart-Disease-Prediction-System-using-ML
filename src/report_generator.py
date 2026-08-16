"""
Report generator for patient risk assessment output (PDF & Text format).
"""

from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

from .config import PRODUCT_NAME, PRODUCT_SUBTITLE, MEDICAL_DISCLAIMER, FEATURE_METADATA, FEATURE_COLUMNS


def generate_pdf_report(
    patient_inputs: Dict[str, Any],
    prediction: int,
    probability: float,
    risk_category: str,
    key_factors: List[Dict[str, Any]],
    model_name: str = "Random Forest Classifier",
    model_metrics: Dict[str, Any] = None
) -> bytes:
    """
    Generates a clean, professional PDF assessment report using ReportLab.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0284c7'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=4
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.HexColor('#991b1b'),
        backColor=colors.HexColor('#fef2f2'),
        borderColor=colors.HexColor('#fca5a5'),
        borderWidth=1,
        borderPadding=8,
        spaceBefore=12,
        spaceAfter=12
    )

    story = []

    # Title & Subtitle Header
    story.append(Paragraph(f"{PRODUCT_NAME} | Assessment Report", title_style))
    story.append(Paragraph(PRODUCT_SUBTITLE, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=15))

    # Assessment Meta Data Table
    assessment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph("<b>Assessment Date:</b>", body_style), Paragraph(assessment_time, body_style),
         Paragraph("<b>Model Name:</b>", body_style), Paragraph(model_name, body_style)],
        [Paragraph("<b>Features Analyzed:</b>", body_style), Paragraph(str(len(FEATURE_COLUMNS)), body_style),
         Paragraph("<b>Assessment Engine:</b>", body_style), Paragraph("CardioAI Analytics v1.0", body_style)]
    ]
    t_meta = Table(meta_data, colWidths=[120, 150, 110, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # Prediction Summary Card Box
    risk_color = colors.HexColor('#ef4444') if prediction == 1 else colors.HexColor('#10b981')
    res_text = f"<b>Predicted Risk Category:</b> <font color='{risk_color.hexval()}'>{risk_category.upper()}</font><br/>" \
               f"<b>Model Probability:</b> {probability * 100:.1f}%<br/>" \
               f"<b>Binary Classification:</b> {'1 (Risk Detected)' if prediction == 1 else '0 (No Disease Risk Detected)'}"

    res_data = [[Paragraph("<font size=12><b>Model Assessment Outcome</b></font>", body_style)],
                [Paragraph(res_text, body_style)]]
    t_res = Table(res_data, colWidths=[540])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f9ff')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#0284c7')),
        ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 15))

    # Patient Inputs Summary Table
    story.append(Paragraph("Patient Clinical Profile", h2_style))
    input_rows = [["Feature Label", "Parameter Name", "Entered Value"]]
    for feat in FEATURE_COLUMNS:
        meta = FEATURE_METADATA.get(feat, {})
        val = patient_inputs.get(feat, "")
        if meta.get("type") == "categorical":
            opts = meta.get("options", {})
            val_str = opts.get(val, str(val))
        else:
            val_str = f"{val} {meta.get('unit', '')}".strip()

        input_rows.append([meta.get("label", feat), feat, val_str])

    t_inputs = Table(input_rows, colWidths=[240, 120, 180])
    t_inputs.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_inputs)
    story.append(Spacer(1, 15))

    # Key Model Influencing Factors
    if key_factors:
        story.append(Paragraph("Key Influencing Model Features", h2_style))
        factor_rows = [["Rank", "Feature Label", "Patient Value", "Relative Model Weight"]]
        for factor in key_factors:
            factor_rows.append([
                str(factor["rank"]),
                factor["label"],
                str(factor["patient_value"]),
                f"{factor['relative_pct']}%"
            ])
        t_factors = Table(factor_rows, colWidths=[40, 240, 160, 100])
        t_factors.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(t_factors)
        story.append(Spacer(1, 15))

    # Medical Disclaimer
    story.append(Paragraph(f"<b>Important Notice:</b> {MEDICAL_DISCLAIMER}", disclaimer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
