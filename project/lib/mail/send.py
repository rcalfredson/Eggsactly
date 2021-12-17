# adapted from this post: https://stackoverflow.com/a/6270987

from email.mime.text import MIMEText
import smtplib


def send_mail(recipients, subject, msg):
    msg = MIMEText(msg)
    msg['Subject'] = subject
    me = 'Egg Count Tester <donotreply@rebeccayang.org>'
    msg['From'] = me
    msg['To'] = ', '.join(recipients)

    # Send the message via our own SMTP server, but don't include the
    # envelope header.
    s = smtplib.SMTP('localhost', 25)
    s.sendmail(me, recipients, msg.as_string())
    s.quit()