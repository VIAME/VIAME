================================
 KWIVER Coding Style Guidelines
================================

.. raw:: html

  <style>
    html {
      color: black;
      background: white;
      font-family: sans;
    }
    p, li { text-align: justify; }
    li { margin-bottom: 1.0em !important; }
    .admonition-rationale {
      background: #def;
      border-radius: 0.5em;
      border: 1px solid #7ae !important;
      margin: 0.5em 1.5em !important;
      padding: 0.5em !important;
    }
    .admonition-example {
      background: #cfd;
      border-radius: 0.5em;
      border: 1px solid #5c7 !important;
      margin: 0.5em 1.5em !important;
      padding: 0.5em !important;
    }
    .admonition-rationale .code,
    .admonition-example .code {
      margin-left: 0 !important;
      padding-left: 0.7em !important;
      border-left: 0.3em solid #aaa;
    }
    h1 > .title {
      font-weight: bold;
      display: block;
      float: right;
    }
    li .title {
      font-weight: bold;
    }
  </style>

.. role:: a(literal)
   :class: title

.. role:: cpp(code)
   :language: c++

.. role:: cmake(code)
   :language: cmake

General Presentation: :a:`[gen]`
''''''''''''''''''''''''''''''''

- :a:`[gen.copyright]`
  Files containing non-trivial, non-machine-generated content should ideally
  include the following copyright notice:

  .. code::

    // This file is part of KWIVER, and is distributed under the
    // OSI-approved BSD 3-Clause License. See top-level LICENSE file or
    // https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

  The comment markers should be replaced according to the type of file.
  (For example, Python code would use :code:`#` rather than :code:`//`.)

- :a:`[gen.len]`
  Prefer to limit lines to at most 79 characters. Use of the 80th character is
  acceptable in cases where doing so is beneficial to readability.

  .. admonition:: Rationale:

    Besides the historic reasons (terminals that are 80 columns in width),
    typographical research has shown that overly long lines are detrimental to
    readability. Lines more than about 80 characters long are more difficult to
    read than lines of more modest length. Additionally, a modest line limit is
    advantageous for viewing multiple files side by side, or in terminal
    windows which are not full screen (even on modern systems, terminal windows
    often default to 80 columns wide) or other contexts such as github diffs
    where long lines are undesirable or problematic.

- :a:`[gen.lines]`
  Place "header lines" immediately before a non-local function definition.
  Header lines may also be placed before other definitions (e.g. classes) where
  deemed beneficial to readability. A header line consists of a C++ comment
  (two forward slashes), followed by a space, followed by a number of hyphens.
  Since it is preferred that the final hyphen be in the 79th column (see
  previous bullet), a header line which starts in the first column (i.e. is not
  indented) should have 76 hyphens.

  .. admonition:: Rationale:

    The use of header lines before functions significantly aids readability.
    The header line is highly visible and *greatly* aids the reader in quickly
    locating function boundaries.

  - :a:`[gen.lines.indent]`
    In case of long function definitions inside of a class definition, consider
    use of indented header lines. An indented header line is indented to the
    same level as the function header, and the number of hyphen characters is
    reduced by the indent level.

- :a:`[gen.regions]`
  Use region markers and separators when appropriate. Region markers consist of
  C++ comments followed *immediately* (i.e. with no intervening whitespace) by
  ``BEGIN`` or ``END``, followed by a brief description of the contents of the
  region. Regions should be separated from one another by a line consisting of
  79 forward slashes surrounded by blank lines. If appropriate, a region
  separator may also separate an unmarked region at the beginning or end of a
  file from an adjacent annotated region. Region begin/end markers should
  always appear in pairs.

  .. admonition:: Rationale:

    Similar to header lines, region markers can help a reader quickly locate
    their place within a file, and also serve to split larger files into
    smaller chunks of related functionality. The specific syntax here described
    is recognized by some editors and will be specially highlighted and may
    create folding regions.

- :a:`[gen.comments]`
  Comments (meaning here, *textual* comments, not region markers and separator
  lines) should be used judiciously. Don't use comments to say something that
  is obvious from the code itself, but do use comments to point out design
  choices, "gotchas", areas that need improvement, and to help separate blocks
  of related code.

  - :a:`[gen.comments.grammar]`
    Comments should start with a capital letter, and should use correct
    spelling and grammar. (If possible, use an editor with built-in spell
    checking.) However, comments normally do *not* end with a period, unless a
    comment consists of more than one sentence. Sentence fragments (as long as
    the grammar is not atrocious) are acceptable.

  - :a:`[gen.comments.format]`
    When using C-style multi-line comments, the initial :cpp:`/*` should be
    indented to the same level as surrounding code. Additional lines should
    start with :code:`*` and be indented one additional space, so that each
    :code:`*` lines up with the *first* :code:`*` of the initial line.
    Note, however, that use of C++ style comments (:cpp:`//`) is preferred.

    .. admonition:: Rationale:

      C++-style comments are slightly easier to type, slightly more terse (no
      trailing :cpp:`*/` line), and, due to each line starting in the same
      column with the same sequence of characters, somewhat less prone to
      formatting butchering by editors. There is also no question of whether to
      start text on the same line as :cpp:`/*` (visually inconsistent) or not
      (another "unneeded" line).

- :a:`[gen.proto]`
  Function prototypes should place the return type, class name, and method name
  on separate lines. Parameters may start on the same line as the method name.

  .. admonition:: Example:

    .. code:: c++

      result_t
      some_class
      ::some_method( int the_param ) const
      {
        ...
      }

- :a:`[gen.class_colon]`
  A :cpp:`:` following a class declaration or constructor should be indented
  and preceded by a newline, if the preceding and following text is not all on
  the same line. (Base class lists, however, may span lines without breaking
  before the :cpp:`:`.)

  .. admonition:: Example:

    .. code:: c++

      // Okay
      class my_class : public really_long_name_of_base_class,
                      protected another_really_long_class_name
      {
        ...
      };

      // Okay
      my_class
      ::my_class() : foo_{ 42 }
      {
      }

      // Also okay
      my_class
      ::my_class()
        : foo_{ 42 }
      {
      }

:a:`[ws]` Whitespace:
'''''''''''''''''''''

- :a:`[ws.tabs]`
  Avoid tabulators.

  .. admonition:: Rationale:

    Rendering of tabulators can be inconsistent, potentially resulting in
    confusing indentation when viewed in a context other than the author's
    editor.

- :a:`[ws.trailing]`
  Avoid trailing whitespace. If possible, configure your editor to
  automatically remove trailing whitespace. This includes unnecessary blank
  lines at the end of a file.

  .. admonition:: Rationale:

    Trailing whitespace almost universally serves no purpose and can contribute
    to unnecessary diff noise. Many tools, including git itself, consider
    trailing whitespace to be an "error" and will highlight it accordingly.

- :a:`[ws.eof]`
  Always end files with a newline character. If possible, configure your editor
  to automatically add a newline if necessary.

  .. admonition:: Rationale:

    Some tools experience confusion or degraded function if a text file does
    not end with a newline character (``cat`` being the canonical example).
    As with trailing whitespace, some tools, including git, consider the lack
    of a terminal newline to be an error.

- :a:`[ws.blanks]`
  Prefer to avoid consecutive blank lines.

  .. admonition:: Rationale:

    Using only single blank lines helps to ensure consistency; it is an easy
    rule to remember, avoiding questions as to when multiple blank lines are
    appropriate. It is also easier to enforce via tools and allows more lines
    of meaningful content to be visible on screen. Proper use of other
    indicators such as header lines generally makes the additional visual
    distinction provided by multiple blank lines unnecessary.

- :a:`[ws.access]`
  Avoid blank lines after an access specifier (e.g. :cpp:`public:`) or the
  :cpp:`case` label of a :cpp:`switch`. However, prefer a blank line *before*
  these, unless the preceding line is the opening :cpp:`{`. (For multiple
  :cpp:`case` labels, omit lines between consecutive labels, placing a blank
  line before the first of the group of labels only.)

- :a:`[ws.space]`
  Use whitespace consistently. KWIVER generally adds whitespace:

  - Inside of matching brackets (all of ``(){}[]<>``).

  - Between a control flow keyword (:cpp:`if`, :cpp:`while`, etc.) and its
    opening parenthesis.

  - On either side of an infix operator,
    including the :cpp:`:` of a range-based :cpp:`for`.

  - Between :cpp:`template` and its opening ``<``.

  - After :cpp:`,` and :cpp:`;`. (However, omit space between consecutive
    :cpp:`;`s, as in e.g. :cpp:`for ( init;; pred )`.)

  Whitespace is normally omitted:

  - Between a prefix or postfix operator and the expression it affects.

  - Between a function/method name and its opening parenthesis.

  - Inside the ``()``\ s of the declaration of a Google Test test case (e.g.
    :cpp:`TEST(suite_name, case_name)`).

    .. This is mainly historic; we may change it at some point, especially if
       we start using automated formatting, since it would otherwise be
       difficult to accomplish.

  Avoid use of more than one space (besides indentation) unless aligning
  related text across multiple lines.

- :a:`[ws.align]`
  Aligning variable or parameter names across multiple lines (i.e. by the use
  of multiple spaces between the type name and identifier) is discouraged.
  (Aligning assignments is usually acceptable.)

- :a:`[ws.namespace]`
  Avoid blank lines in between the opening and closing lines of namespaces.
  *Do* use a blank line between the opening of a namespace and any contents of
  that namespace other than a nested namespace, and between the end of such
  content and the brace closing the namespace.

  .. admonition:: Example:

    .. code:: c++

      namespace kwiver {
      namespace vital {

      struct some_type
      {
        ...
      };

      } // namespace vital
      } // namespace kwiver

:a:`[indent]` Indentation and Braces:
'''''''''''''''''''''''''''''''''''''

- :a:`[indent.amount]`
  Use two spaces per level to indent.

- :a:`[indent.broken]`
  Indent lists starting on the next line by one level relative to the list
  scope.

  .. admonition:: Example:

    .. code:: c++

      auto var = this_is_a_long_function(
                   it_has_many_parameters, that_have_very_long_names,
                   which_do_not_fit_on_one_line);

- :a:`[indent.continuation]`
  Indent broken lists to the same indentation as the first item.

  .. admonition:: Example:

    .. code:: c++

      example(this_function_also_has_a_really_long_parameter_list,
              so_it_too_needs_to_span_multiple_lines);

- :a:`[indent.operator]`
  Prefer to break *after* operators, rather than before.

  .. admonition:: Rationale:

    Lines starting with operators tend to align in a way that is not
    aesthetically pleasing. Breaking after the operator rather than before is
    often more readable, and also serves as an indication that the code
    continues on the next line.

    .. code:: c++

      // This looks strange
      if (this_is_some_really_long_condition
          && this_is_another_really_long_condition)

      // This looks better; the conditions are aligned
      if (this_is_some_really_long_condition &&
          this_is_another_really_long_condition)

  - Exception: break *before* the :cpp:`<<` and :cpp:`>>` stream operators, and
    align the first operator of a new line with the first use of the operator.

    .. admonition:: Example:

      .. code:: c++

        std::cout << "This really long line at " << __LINE__
                  << "needs to be split";

        EXPECT_EQ(long_name, another_long_name)
          << "My assertion message does not fit on the same line!";

- :a:`[indent.braces]`
  Use `Allman Style`_ braces. Indent braces to the same level as their
  enclosing scope and/or initiating statement. Place initial braces on a new
  line.

  - :a:`[indent.braces.lambda]`
    As an exception to the above, the initial brace of an initializer list or
    lambda normally should *not* start a new line.

  - :a:`[indent.braces.optional]`
    Prefer to use braces around single-line statements.

- :a:`[indent.namespace]`
  Do not indent contents of namespaces.

- :a:`[indent.trailing_return]`
  Do not indent the :cpp:`->` of a trailing return type specifier; this should
  instead line up with the function name.

  .. admonition:: Example:

    .. code:: c++

      auto
      my_function( ... )
      -> decltype( ... );

Type Names: :a:`[types]`
''''''''''''''''''''''''

- :a:`[types.qualified]`
  Prefer :cpp:`T const` to :cpp:`const T`.

  .. admonition:: Rationale:

    In all cases except a left-most :cpp:`const`, the :cpp:`const` modifier
    affects the type which immediately precedes it. By always placing
    :cpp:`const` to the right, the exceptional case is avoided, thus reducing
    potential confusion as to what the :cpp:`const` is modifying. If the
    modified type is an alias, this can avoid confusion such as mistaking
    :cpp:`const T_PTR` for :cpp:`const T*`, when it is actually
    :cpp:`T* const`. (At least one (non-Kitware) library has a thoroughly wonky
    C API due to this exact mistake!)

    As another example, consider doing an automated find-and-replace to change
    ``T*`` to ``T_ptr``. With ``const T*``, this will be dangerous, as it can
    result in a change of type that is not intended, where ``T const*`` will
    not match the naïve replacement pattern and will thus force the developer
    to consider the appropriate replacement for that case.

  .. admonition:: Tip:

    The regular expression ``const ([\w:]+(<[^>]+>)?)(?! *\w+ *=)`` can be used
    to find and replace many instances of :cpp:`const T`, using the replacement
    template ``\1 const``. Note, however, that this will not work correctly for
    :cpp:`T` which has nested template types, nor has it been rigorously tested
    against false positives. Use with caution and be sure to review all changes
    that are made.

- :a:`[types.auto]`
  Prefer to use :cpp:`auto`, especially for overly long type names and where
  the type is obvious from context. *Especially* prefer to use :cpp:`auto` if
  the type name is already present on the RHS of an assignment (such as when
  the RHS is a :cpp:`static_cast`).

  .. admonition:: Rationale:

    Appropriate use of :cpp:`auto` reduces clutter and can allow for easier
    refactoring, as well as ensuring that variables are initialized. In most
    cases, the actual type is not critical to the correct implementation of an
    algorithm; only that the *appropriate* type (which can be derived using
    :cpp:`auto`) is used. Even in cases where a specific type must be named,
    the name can almost always be written on the RHS of an assignment. See Herb
    Sutter's |gotw94|_ for more details.

    Most modern IDE's can deduce (and display) actual types when :cpp:`auto` is
    used for those instances when a reader needs to know the actual type.

- :a:`[types.const]`
  Prefer to :cpp:`const`-qualify variables whenever possible. Additionally,
  prefer to make literal constants (that is, identifiers whose value is
  statically known) :cpp:`constexpr`.

  .. admonition:: Rationale:

    Making variables immutable helps to avoid unintended modification, and may
    permit additional compiler optimizations.

- :a:`[types.aliases]`
  Create type aliases where appropriate. In particular, prefer to use type
  aliases in class definitions to clarify the intent of a specific
  instantiation of a template type.

Includes: :a:`[include]`
''''''''''''''''''''''''

- :a:`[include.groups]`
  Separate groups of include directives with a single blank line. A "group" is
  a set of headers which belong to the same library or module.

  .. admonition:: Rationale:

    Keeping groups separate improves readability and is necessary for other
    include rules to be applied sensibly.

- :a:`[include.group_order]`
  Order groups of includes in decreasing order of dependency. The header
  corresponding to the source file (e.g. ``#include "foo.h"`` in ``foo.cpp``)
  should always be first. (Private headers, e.g. ``foo_priv.h``, should appear
  before ``foo.h``, or instead of ``foo.h`` if that is included by the private
  header.) Local headers should follow. Low level (e.g. POSIX) headers should
  appear last, preceded by C portability headers (e.g. :cpp:`<cmath>`),
  preceded by Standard Library headers (e.g. :cpp:`<memory>`).

  .. admonition:: Rationale:

    This ordering helps to detect if a header fails to include the headers of
    lower level components on which it depends, by reducing the likelihood that
    such lower level headers have been previously included. In particular,
    including the public header for a particular component first in that
    component's source file helps to ensure that the component's header is
    "self contained".

- :a:`[include.order]`
  Prefer to order includes within a group by lexicographical order. (Don't get
  hung up on the correct order of symbols versus letters, however, so long as
  such ordering is consistent within a group.)

  .. admonition:: Rationale:

    Within a group, the ability to infer order of dependency is typically
    limited; thus, some other criteria is needed to keep includes from being in
    arbitrary order. Lexicographical order is easy to remember.

Miscellaneous: :a:`[misc]`
''''''''''''''''''''''''''

- :a:`[misc.modern]`
  Use modern C++ when possible and applicable. In particular:

  - :a:`[misc.modern.range_for]`
    Prefer to use range-based :cpp:`for`.

    .. admonition:: Example:

      .. code:: c++

        // Ugly
        for ( metadata_map::iterator iter = md.begin();
              iter != md.end(); ++iter )

        // Much better
        for ( auto item : md )

        // If you really need the iterator...
        for ( auto iter : md | kwiver::vital::range::indirect )

  - :a:`[misc.modern.typedef]`
    Write type aliases like :cpp:`using alias_name = aliased_type`.
    Avoid :cpp:`typedef`.

  - :a:`[misc.modern.nullptr]`
    Always write :cpp:`nullptr`. Never use :cpp:`0` as a pointer.

  - :a:`[misc.modern.override]`
    Always decorate virtual method overrides with :cpp:`override`.
    Use of the :cpp:`virtual` keyword is discouraged in declarations
    with :cpp:`override`.

  - :a:`[misc.modern.member_init]`
    Prefer inline member initialization when possible.

    .. admonition:: Example:

      .. code:: c++

        // Pre-C++11
        struct foo
        {
          Foo() : bar(42) {}
          int bar;
        };

        // C++11
        struct Foo
        {
          Foo() {}
          int bar = 42;
        };

  - :a:`[misc.modern.construct]`
    Prefer uniform initialization (using ``{}``\ s, not ``()``\ s).

  - :a:`[misc.modern.elision]`
    Prefer to omit unneeded type names when constructing objects inline.

    .. admonition:: Example:

      .. code:: c++

        Foo bar()
        {
          none({42}); // Parameter type name elided
          return {42}; // Return value type name elided
        };

  .. admonition:: Rationale:

    Besides being "more modern" for its own sake, modern C++ tends to be easier
    to read and understand with less unnecessary clutter, and in some cases,
    expresses programmer intent more explicitly, which allows the compiler to
    catch more errors.

- :a:`[misc.postfix]`
  Avoid use of postfix increment and decrement unless the old value is needed.

  .. admonition:: Rationale:

    Since postfix increment/decrement returns the *old* value, while prefix
    increment/decrement returns the *new* value, the implementation of the
    latter is usually more efficient. While this may not matter for integer
    data types (assuming that the compiler will optimize away the unneeded code
    when it sees that the result is unused), it is good to be consistent.

- :a:`[misc.new]`
  Avoid :cpp:`new` when possible. In particular, avoid :cpp:`new` when creating
  a :cpp:`shared_ptr`; use :cpp:`make_shared` instead.

  .. admonition:: Rationale:

    Using :cpp:`make_shared` reduces repetition; combined with :cpp:`auto`, in
    most cases the type name will only appear once. More importantly, however,
    it is more efficient in many cases. For a more detailed rationale, see Herb
    Sutter's |gotw89|_.

- :a:`[misc.casts]`
  Avoid explicit casts when an implicit conversion will suffice. In particular,
  avoid use of :cpp:`const_cast` and :cpp:`const_pointer_cast`, which are
  usually indicators that a potentially dangerous operation is occurring, to
  *add* :cpp:`const`-qualification; this can almost always be done implicitly.

- :a:`[misc.locals]`
  Prefer to store intermediate values in local (:cpp:`const`-qualified!)
  variables. This increases the chances of being able to inspect these values
  in a debugger.

- :a:`[misc.include_guard]`
  Prefer to omit comments after the :cpp:`#endif` of a multiple-inclusion
  guard.

  .. admonition:: Rationale:

    Although it is fairly common practice to repeat the guard symbol after the
    :cpp:`#endif`, these comments actually serve very little purpose, and they
    add an additional maintenance burden. Headers are often copied or renamed,
    and it is very easy for these comments to become outdated and incorrect.

    Although we *do* recommend similar comments after the brace ending a
    namespace, namespaces change far less often, and a single brace is much
    more ambiguous, especially as namespaces may be nested and/or end in the
    middle of a file, whereas multiple-inclusion guards are never nested and
    the :cpp:`#endif` is almost universally the last line of the header.

API Style: :a:`[api]`
'''''''''''''''''''''

- :a:`[api.naming]`
  Prefer to follow STL naming conventions (lower case names with ``_`` between
  words) for symbol names.

- :a:`[api.abbrev]`
  Avoid the use of abbreviations in names, especially in public API. Acronyms,
  especially where the full phrase is rarely or almost never used (e.g. "IO",
  "URI"), are okay, but prefer to use the full phrase if in doubt. (As an
  exception, :cpp:`foo_sptr` and :cpp:`foo_scptr` are commonly used to denote
  a :cpp:`shared_ptr` to a :cpp:`foo` or :cpp:`foo const`, respectively.)

  .. admonition:: Rationale:

    The use of abbreviations is detrimental to the accessibility of an API, as
    it is difficult for users to remember when a term is abbreviated and, in
    some cases, how (for example, was that method named "cur_frame",
    "curr_frame" or "current_frame"?). Avoiding abbreviations avoids this
    confusion, results in clearer code (since the reader doesn't have to stop
    to puzzle out what the abbreviation means), and encourages greater care to
    be given to devising concise names.

- :a:`[api.return]`
  Prefer to avoid returning references. There may be exceptions where returning
  a reference is necessary, but in general it is dangerous as it opens the
  possibility of the reference outliving its owner. Moreover, if you *must*
  return a reference to an object your class owns, *strongly* consider adding
  an r-value qualified overload of the method in question that either returns a
  copy or is explicitly deleted, so that callers cannot accidentally call a
  reference-returning method on a temporary instance of your class.

  .. admonition:: Rationale:

    It is a common idiom (see |gotw88|) to assign a result from a method to a
    :cpp:`const&`-qualified local variable. This is an old (and to be fair,
    probably no longer necessary) trick to avoid an unnecessary copy. However,
    if the method in question returns a *real* reference, it becomes a disaster
    waiting to happen if the owner of the reference goes out of scope before
    the local variable, especially if the reference is owned by the object on
    which the method is called, and that method is called on a temporary.

- :a:`[api.pimpl]`
  Use PIMPL_ when appropriate.

  .. admonition:: Example:

    .. code:: c++

      class foo
      {
        // ...

      protected:
        class priv;
        std::unique_ptr< priv > const d;
      };

- :a:`[api.export]`
  Remember to decorate symbols that should be exported. Use generated export
  headers.

- :a:`[api.exceptions]`
  Use exceptions and return values, not error codes and output parameters.

  .. admonition:: Rationale:

    This allows for chaining functions, works with ``<algorithm>`` better,
    and allows more variables to be :cpp:`const`.

API Documentation: :a:`[doc]`
'''''''''''''''''''''''''''''

C++ code is documented using Doxygen. Strive to provide documentation for all
public interfaces (free functions, member functions, and member variables)
where such documentation is not superfluous.

- :a:`[doc.format]`
  Use :cpp:`///` for all doxygen documentation, except for those directives
  which require other syntax (e.g. :cpp:`//@{` and :cpp:`//@}`). Avoid use of
  C-style Doxygen comments. (See ``[gen.comments.format]`` for rationale.)

- :a:`[doc.directives]`
  Use ``\command`` rather than ``@command``.

- :a:`[doc.keywords]`
  When a code keyword (e.g. :cpp:`true`) or parameter name appears in
  documentation, prefer to annotate these with ``\c`` or ``\p``, respectively.

- :a:`[doc.periods]`
  End all documentation with a period, including brief documentation consisting
  of a single sentence or sentence fragment. Use a single space after
  sentences.

  .. admonition:: Rationale:

    Always ending documentation with periods is fairly common practice, and
    improves consistency by ensuring that all documentation renders with a
    terminal period, both in the generated HTML and in the corresponding
    source. Using a single space after sentences, rather than two, is much
    easier to tool-enforce.

- :a:`[doc.superfluous]`
  Omit superfluous documentation. In particular, avoid documentation that
  simply reiterates the name of the thing being documented.

  .. admonition:: Rationale:

    Documenting a copy constructor as "copy constructor" is essentially a waste
    of space. Just as excessive use of code comments to state the obvious is
    discouraged, we also wish to avoid documentation that is effectively
    gratuitous.

    Note that this does not apply to brief descriptions when additional,
    meaningful detailed documentation is also present.

- :a:`[doc.wordiness]`
  Avoid unnecessarily "wordy" brief documentation. In particular, avoid
  starting brief descriptions with unnecessary articles ("a", "the", etc.).
  (Note that this does *not* apply to detailed descriptions.)

- :a:`[doc.brief]`
  Make use of AUTOBRIEF. Avoid use of ``\brief`` and write the first paragraph
  of documentation so that it is suitable as a brief description. Keep in mind
  that the entire first paragraph of documentation will be used; you are not
  limited to one line or one sentence. Make sure to start a new paragraph
  before detailed documentation.

- :a:`[doc.blanks]`
  Always insert a blank line between a (previous, unrelated) declaration and
  documentation. Avoid a blank line between documentation and the code being
  documented.

- :a:`[doc.repeating]`
  Avoid repeating documentation from a header (``.h``) in an implementation
  (``.cpp``).

  .. admonition:: Rationale:

    Repeating documentation increases the size of implementations and adds
    maintenance burden, and it is arguable whether or not doing so adds any
    meaningful benefit.

Unit Tests: :a:`[test]`
'''''''''''''''''''''''

New code should have unit tests wherever possible.
Google Test is used for writing C++ unit tests.

- :a:`[test.naming]`
  Tests should use a suite name that reflects the class or algorithm being
  tested, and a case name that reflects what aspect or behavior of the class or
  algorithm is being tested. See existing tests for examples.

- :a:`[test.parameterized]`
  Prefer to use parameterized tests when appropriate. Avoid creating multiple
  test cases that differ only by input types or values. Also avoid setting up
  test cases where a single test case performs the same set of tests on a set
  of types or values; these should be refactored as parameterized test.

- :a:`[test.reuse]`
  Reuse test code when possible. If two or more arrows implement similar
  algorithms, try to implement the tests so that they share code as much as
  possible. See ``arrows/tests`` for some examples.

- :a:`[test.assertions]`
  Don't forget to use fatal assertions (``ASSERT_*`` vs. ``EXPECT_*``) when
  appropriate. If it does not make sense to continue a test case after a
  particular failure, use a fatal rather than non-fatal assertion. Especially
  use fatal assertions when obtaining a resource for later use in order to
  prevent attempted use of a non-existing resource from causing a null pointer
  dereference. (Similarly, don't use :cpp:`if` to avoid crashes that are better
  prevented by stopping a test case via a fatal assertion.)

- :a:`[test.helpers]`
  Use helper functions or inline, immediately invoked lambdas when it is
  helpful for an assertion to terminate a block of code, but not the entire
  test case.

  .. admonition:: Example:

    .. code:: c++

      TEST(foo, bar)
      {
        // Abort loop (but not test case) on first failed point
        auto points = compute_points();
        [&]{
          for ( auto const& p : points )
          {
            ASSERT_TRUE( test_point( p ) );
          }
        }();

        // More assertions...
      }

- :a:`[test.trace]`
  Make use of ``SCOPED_TRACE``. Using this to provide information about the
  current loop iteration inside of a loop body is especially useful.

- :a:`[test.info]`
  Provide additional information about a failed assertion when necessary, but
  do so *judiciously*. In particular, resist the urge to repeat information
  that is already available from the assertion itself. Keep in mind that Google
  Test will print the arguments of a failed assertion as well as the location
  of the failure.

  .. admonition:: Example:

    .. code:: c++

      TEST(foo, bar)
      {
        // Completely redundant; don't do this!
        EXPECT_EQ( b, a ) << "a should be equal to b";

        // Redundant; prefer to not do this
        EXPECT_EQ( s.good() ) << "Stream is good before seek";
        s.seek( ... );

        // Better
        auto r = get_resource();
        ASSERT_TRUE( !!r ) << "Failed to obtain required resource";
      }

CMake: :a:`[cmake]`
'''''''''''''''''''

To the extent possible, CMake source should follow the same rules as C++ code.
In particular:

- :a:`[cmake.indent]`
  Use two spaces to indent.

- :a:`[cmake.line_breaks]`
  Break lines in the same manner as in C++.

Also, try to follow best practices for modern CMake, and use KWIVER utility
functions as appropriate. In particular:

- :a:`[cmake.variables]`
  Use lowercase for private variables, and uppercase for user-controlled
  variables.

- :a:`[cmake.functions]`
  Prefer functions over macros

  .. admonition:: Rationale:

    Unlike macros, functions create a new variable scope which prevents
    "leaking" variables into the caller's scope. They are also easier to debug.

- :a:`[cmake.lists]`
  Prefer :cmake:`foreach (var IN LISTS list)` and :cmake:`list(APPEND)`.

- :a:`[cmake.configure_file]`
  Prefer :cmake:`kwiver_configure_file` over :cmake:`configure_file` when
  possible.

  .. admonition:: Rationale:

    :cmake:`kwiver_configure_file` sets up a custom command to generate the
    configured file at build time, rather than at configure time. This reduces
    the configure dependencies and avoids forcing the user to re-run CMake when
    the inputs change.

- :a:`[cmake.wrappers]`
  Use the ``kwiver_`` wrappers of common commands (e.g., :cmake:`add_library`,
  :cmake:`add_test`, etc.) as they automatically Do The Right Thing with
  installation, compile flags, build locations, and more.

- :a:`[cmake.paths]`
  Quote *all* paths and variable expansions unless list expansion is required
  (usually in command arguments or optional arguments).

.. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..

.. _Allman Style: https://en.wikipedia.org/wiki/Indent_style#Allman_style

.. _gotw88: https://herbsutter.com/2008/01/01/gotw-88-a-candidate-for-the-most-important-const/

.. _gotw89: https://herbsutter.com/2013/05/29/gotw-89-solution-smart-pointers/

.. _gotw94: https://herbsutter.com/2013/08/12/gotw-94-solution-aaa-style-almost-always-auto/

.. _PIMPL: https://en.wikipedia.org/wiki/Opaque_pointer

.. |gotw88| replace:: GOTW #88

.. |gotw89| replace:: GOTW #89

.. |gotw94| replace:: GOTW #94
