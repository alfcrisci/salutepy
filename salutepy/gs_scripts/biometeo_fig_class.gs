function main(args)
         file = subwrd(args,1)
         variable = subwrd(args,2)
         id = subwrd(args,3) 

say 'file =' file
say 'variable = ' variable
say 'id = ' id

'sdfopen 'file
'set mpt * off'
'set gxout shaded'
'set csmooth on'
'set map 1 1 1'
'set parea 0.95 8.04 0.95 8.04'
'set grads off'
'set grid off'
'set lon 6 19'
'set lat 36.5 47.5'
'run legend/palette/palette_'variable'.gs'
'd 'variable
'sdfopen legend/mask/mask_0025.nc'
'set dfile 2'
'set lon 6 19'
'set lat 36.5 47.5'
'set gxout fgrid'
'set z 1'
'set t 1'
'set fgvals 1 0'
'd mask_rho.2'
'set dfile 1'
'set mpdset legend/maps/freegis_countries legend/maps/regioni'
'set mpt * 1 1 1'
'draw map'
'run legend/strings/'variable'_string.gs'
'run legend/code/cbarn_'variable'.gs 1 0 4.5 0.35'
'set font 0'
'set string 15 tl 4'
'set strsiz 0.13'
'printim 'variable'_'id'.png x730 y600 white'
'quit'
