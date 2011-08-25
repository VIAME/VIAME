" Vim syntax file
" Language:	vistk pipeline
" Maintainer:	Ben Boeckel <ben.boeckel@kitware.com>
" Last Change:	2011-08-24
" Credits:	Ben Boeckel <ben.boeckel@kitware.com>

if exists("b:current_syntax")
    finish
endif

syn case match

syn match pipeConnectDecl /\(^ *\)\zs\<connect\>/ nextgroup=pipeConnectFrom skipwhite
syn match pipeConnectFrom /\<from\>/              nextgroup=pipeConnectPortAddr contained skipwhite
syn match pipeConnectTo   /\(^ *\)\zs\<to\>/      nextgroup=pipePortAddr contained skipwhite

syn match pipeImapDecl /\(^ *\)\zs\<imap\>/ nextgroup=pipeIMapFlags skipwhite
syn match pipeImapFrom /\<from\>/           nextgroup=pipeIMapPortAddr skipwhite
syn match pipeImapTo   /\(^ *\)\zs\<to\>/   nextgroup=pipePort skipwhite

syn match pipeOmapDecl /\(^ *\)\zs\<omap\>/ nextgroup=pipeOMapFlags skipwhite
syn match pipeOmapFrom /\<from\>/           nextgroup=pipeOMapPort skipwhite
syn match pipeOmapTo   /\(^ *\)\zs\<to\>/   nextgroup=pipePortAddr skipwhite

syn match pipeConnectPortAddr /[a-zA-Z_-]\+\.[a-zA-Z_-]\+$/ nextgroup=pipeConnectTo contained skipnl skipwhite

syn match pipeImapFlags    /\(\[[a-zA-Z_-]\+\(,[a-zA-Z_-]\+\)*\]\)?/ nextgroup=pipeImapFrom contained skipwhite
syn match pipeImapPortAddr /[a-zA-Z_-]\+$/                           nextgroup=pipeImapTo contained skipnl skipwhite

syn match pipeOmapFlags /\(\[[a-zA-Z_-]\+\(,[a-zA-Z_-]\+\)*\]\)?/ nextgroup=pipeOmapFrom contained skipwhite
syn match pipeOmapPort  /[a-zA-Z_-]\+\.[a-zA-Z_-]\+$/             nextgroup=pipeOmapTo contained skipnl skipwhite

syn match pipePort     /[a-zA-Z_-]\+$/               contained
syn match pipePortAddr /[a-zA-Z_-]\+\.[a-zA-Z_-]\+$/ contained

syn keyword pipeTodo  FIXME NOTE NOTES TODO XXX contained
syn match pipeComment /\(^ *\)\zs#.*/           contains=pipeTodo,@Spell

syn match pipeInclude     /^!include/         nextgroup=pipeIncludeFile skipwhite
syn match pipeIncludeFile /[a-zA-Z_/.\\-]\+$/ contained

syn match pipeBlockDecl  /\(^ *\)\zs\<\(process\|group\)\>/    nextgroup=pipeName skipwhite
syn match pipeConfigDecl /\(^ *\)\zs\<config\>/                nextgroup=pipeConfig skipwhite
syn match pipeName       /\<[a-zA-Z_-]\+\>$/                   contained
syn match pipeConfig     /\<[a-zA-Z_-]\+\(:[a-zA-Z_-]\+\)*\>$/ contained

syn match pipeConfigType /\(^ *\)\zs::/  nextgroup=pipeType skipwhite
syn match pipeType       /[a-zA-Z_-]\+$/ contained

syn match pipeConfigIndex    /\(^ *\)\zs\(:[a-zA-Z_-]\+\)\+/            nextgroup=pipeConfigFlags
syn match pipeConfigFlags    /\(\[[a-zA-Z_-]\+\(,[a-zA-Z_-]\+\)*\]\)\?/ nextgroup=pipeConfigProvider contained
syn match pipeConfigProvider /\({[A-Z]\+}\)\?/                          nextgroup=pipeConfigValue contains=pipeConfigValue contained skipwhite
syn match pipeConfigValue    /\([a-zA-Z0-9\._/: -]\+\|".*"\)$/          contains=@Spell contained

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
