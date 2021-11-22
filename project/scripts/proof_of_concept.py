import os
import sys
sys.path.append(os.path.abspath("./"))
from lib.mail.send import send_mail

send_mail(['rca30@duke.edu'], 'Sending via Python', 'testing. line1\nline2\netc.')