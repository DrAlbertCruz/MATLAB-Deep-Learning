function t_mailtest( m_text, m_subject )
if nargin == 0
    m_subject = 'VICTORY JOB SERVER: Error';
    m_text = 'Victory has had an error';
end

mail = 'pikez33@yahoo.com';
psswd = 'xBjQ^17I3EM!xbAz6#zc';
host = 'smtp.mail.yahoo.com';
port  = '465';

emailto = 'cruz0301@gmail.com';

setpref( 'Internet','E_mail', mail );
setpref( 'Internet', 'SMTP_Server', host );
setpref( 'Internet', 'SMTP_Username', mail );
setpref( 'Internet', 'SMTP_Password', psswd );

props = java.lang.System.getProperties;
props.setProperty( 'mail.smtp.user', mail );
props.setProperty( 'mail.smtp.host', host );
props.setProperty( 'mail.smtp.port', port );
props.setProperty( 'mail.smtp.starttls.enable', 'true' );
props.setProperty( 'mail.smtp.debug', 'true' );
props.setProperty( 'mail.smtp.auth', 'true' );
props.setProperty( 'mail.smtp.socketFactory.port', port );
props.setProperty( 'mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory' );
props.setProperty( 'mail.smtp.socketFactory.fallback', 'false' );

sendmail( emailto , m_subject, m_text );