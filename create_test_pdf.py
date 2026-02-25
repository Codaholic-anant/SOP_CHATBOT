from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("terms.pdf")
elements = []

styles = getSampleStyleSheet()

content = """
Company SOP - Test Document

1. Refund Policy:
Customers can request a refund within 7 days of purchase.

2. Leave Policy:
Employees are allowed 20 paid leaves per year.

3. Working Hours:
Office hours are 9 AM to 6 PM, Monday to Friday.

4. Remote Work:
Remote work is allowed with manager approval.

5. Support Contact:
For issues, contact support@company.com
"""

elements.append(Paragraph(content, styles["Normal"]))

doc.build(elements)

print("Test PDF created successfully!")