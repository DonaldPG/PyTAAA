def SendEmail( username, emailpassword, toperson, fromperson, subjecttext, regulartext, boldtext, headlinetext ) :
    import smtplib
    # to, from, subject
    fromaddr = fromperson
    toaddrs  = toperson
    message_from = "From: <"+fromaddr+">"
    message_to   = "To: <"+toaddrs+">"
    message_subject = subjecttext

    # message body
    message = message_from+"\n"+message_to+"\n"+"""MIME-Version: 1.0
Content-type: text/html
Subject: """+message_subject+"""

<h1>"""+headlinetext+"""</h1>
<b>"""+boldtext+"""</b>

"""+regulartext+"""

"""

    # The actual mail send
    try:
        server = smtplib.SMTP('smtp.gmail.com:587')
        server.starttls()
        server.login(username,emailpassword)
        server.sendmail(fromaddr, toaddrs, message)
        server.quit()
        #print "email message = "
        #print message
        print "Successfully sent email"
        print ""
    except :
        print "Error: unable to send email"
        print ""


