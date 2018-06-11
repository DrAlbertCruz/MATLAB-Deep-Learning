try
	t_ashrafGrapesGoogLeNet( 1:10 )
	t_italyGrapesGoogLeNet( 1:10 )
	t_ashrafGrapesGoogLeNet( 11:60 )
	t_italyGrapesGoogLeNet( 11:60 )
catch e
    t_mailtest();
    e
end