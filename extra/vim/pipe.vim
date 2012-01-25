" Vim syntax file
" Language:	vistk pipeline
" Maintainer:	Ben Boeckel <ben.boeckel@kitware.com>
" Last Change:	2012-01-25
" Credits:	Ben Boeckel <ben.boeckel@kitware.com>

if exists("b:current_syntax")
    finish
endif

syn case match

let s:begin_line='\(^[ \t]*\)'
let s:config_name='[a-zA-Z0-9_-]\+'
let s:port_name='[a-zA-Z0-9_/-]\+'
let s:flag='[a-zA-Z]\+'
let s:provider='[A-Z]\+'
let s:config_value='[a-zA-Z0-9./:_ \t-]\+'

exec 'syn match pipeConnectDecl /' . s:begin_line . '\zs\<connect\>/ nextgroup=pipeConnectFrom               skipwhite'
exec 'syn match pipeConnectFrom                        /\<from\>/    nextgroup=pipeConnectPortAddr contained skipwhite'
exec 'syn match pipeConnectTo   /' . s:begin_line . '\zs\<to\>/      nextgroup=pipePortAddr        contained skipwhite'

exec 'syn match pipeImapDecl /' . s:begin_line . '\zs\<imap\>/ nextgroup=pipeIMapFlags           skipwhite'
exec 'syn match pipeImapFrom                        /\<from\>/ nextgroup=pipeIMapPort  contained skipwhite'
exec 'syn match pipeImapTo   /' . s:begin_line . '\zs\<to\>/   nextgroup=pipePortAddr  contained skipwhite'

exec 'syn match pipeOmapDecl /' . s:begin_line . '\zs\<omap\>/ nextgroup=pipeOMapFlags              skipwhite'
exec 'syn match pipeOmapFrom                        /\<from\>/ nextgroup=pipeOMapPortAddr contained skipwhite'
exec 'syn match pipeOmapTo   /' . s:begin_line . '\zs\<to\>/   nextgroup=pipePort         contained skipwhite'

exec 'syn match pipeConnectPortAddr /' . s:config_name . '\.' . s:port_name . '$/ nextgroup=pipeConnectTo contained skipnl skipwhite'

exec 'syn match pipeImapFlags /\(\[' . s:flag . '\(,' . s:flag . '\)*\]\)\?/ nextgroup=pipeImapFrom contained        skipwhite'
exec 'syn match pipeImapPort  /' . s:port_name . '$/                         nextgroup=pipeImapTo   contained skipnl skipwhite'

exec 'syn match pipeOmapFlags    /\(\[' . s:flag . '\(,' . s:flag . '\)*\]\)\?/ nextgroup=pipeOmapFrom contained        skipwhite'
exec 'syn match pipeOmapPortAddr /' . s:config_name . '\.' . s:port_name . '$/  nextgroup=pipeOmapTo   contained skipnl skipwhite'

exec 'syn match pipePort     /' . s:port_name . '$/                        contained'
exec 'syn match pipePortAddr /' . s:config_name . '\.' . s:port_name . '$/ contained'

exec 'syn keyword pipeTodo  FIXME NOTE NOTES TODO XXX                             contained'
exec 'syn match pipeComment /' . s:begin_line . '\zs#.*/ contains=pipeTodo,@Spell'

exec 'syn match pipeInclude     /^!include/       nextgroup=pipeIncludeFile           skipwhite'
exec 'syn match pipeIncludeFile /[a-zA-Z_/.-]\+$/                           contained'

exec 'syn match pipeBlockDecl  /' . s:begin_line . '\zs\<\(process\|group\)\>/         nextgroup=pipeName             skipwhite'
exec 'syn match pipeConfigDecl /' . s:begin_line . '\zs\<config\>/                     nextgroup=pipeConfig           skipwhite'
exec 'syn match pipeName       /\<' . s:config_name . '\>$/                                                 contained'
exec 'syn match pipeConfig     /\<' . s:config_name . '\(:' . s:config_name . '\)*\>$/                      contained'

exec 'syn match pipeConfigType /' . s:begin_line . '\zs::/  nextgroup=pipeType           skipwhite'
exec 'syn match pipeType       /' . s:config_name . '$/                        contained'

exec 'syn match pipeConfigIndex    /' . s:begin_line . '\zs\(:' . s:config_name . '\)\+/ nextgroup=pipeConfigFlags'
exec 'syn match pipeConfigFlags    /\(\[' . s:flag . '\(,' . s:flag . '\)*\]\)\?/        nextgroup=pipeConfigProvider contained'
exec 'syn match pipeConfigProvider /\({' . s:provider . '}\)\?/                          nextgroup=pipeConfigValue    contained skipwhite'
exec 'syn match pipeConfigValue    /' . s:config_value . '$/                             contains=@Spell              contained'

hi def link pipeDecl              Keyword
hi def link pipeConn              Character
hi def link pipeFlags             Number
hi def link pipeProvider          StorageClass
hi def link pipeBlockName         Function
hi def link pipeKey               Identifier
hi def link pipeValue             Constant
hi def link pipeType              Type
hi def link pipeAddr              Statement

hi def link pipeTodo              Todo
hi def link pipeComment           Comment

hi def link pipeInclude           Include
hi def link pipeIncludeFile       String

hi def link pipeConnectDecl       pipeDecl
hi def link pipeImapDecl          pipeDecl
hi def link pipeOmapDecl          pipeDecl
hi def link pipeBlockDecl         pipeDecl
hi def link pipeConfigDecl        pipeDecl

hi def link pipeConnectTo         pipeConn
hi def link pipeConnectFrom       pipeConn
hi def link pipeImapTo            pipeConn
hi def link pipeImapFrom          pipeConn
hi def link pipeOmapTo            pipeConn
hi def link pipeOmapFrom          pipeConn

hi def link pipeImapFlags         pipeFlags
hi def link pipeOmapFlags         pipeFlags

hi def link pipeConnectPortAddr   pipeAddr
hi def link pipeImapPort          pipeAddr
hi def link pipeOmapPortAddr      pipeAddr
hi def link pipePort              pipeAddr
hi def link pipePortAddr          pipeAddr

hi def link pipeName              pipeBlockName
hi def link pipeConfig            pipeBlockName

hi def link pipeConfigFlags       pipeFlags
hi def link pipeConfigProvider    pipeProvider

hi def link pipeConfigType        pipeValue
hi def link pipeType              pipeType

hi def link pipeConfigIndex       pipeKey
hi def link pipeConfigValue       pipeValue

let b:current_syntax = "pipe"
