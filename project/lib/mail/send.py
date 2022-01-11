# adapted from this post: https://stackoverflow.com/a/6270987

from email.mime.text import MIMEText
import json
import os
import smtplib

if os.getenv("FLASK_ENV") == "development":
    email_list_suffix = "_dev"
elif os.getenv("FLASK_ENV") == "production":
    email_list_suffix = ""
with open(f"notification_emails{email_list_suffix}.json") as f:
    default_recipients = json.load(f)


def send_mail(subject, msg, recipients=default_recipients):
    msg = MIMEText(msg)
    msg["Subject"] = subject
    me = "Egg Count Tester <donotreply@rebeccayang.org>"
    msg["From"] = me
    msg["To"] = ", ".join(recipients)

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP("localhost", 25)
    s.sendmail(me, recipients, msg.as_string())
    s.quit()
