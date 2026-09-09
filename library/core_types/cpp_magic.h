/**
 * LICENSE
 * SOURCE: https://github.com/18sg/uSHET/blob/master/lib/cpp_magic.h
 *
 * uSHET Library
 * =============
 *
 * Copyright (c) 2014 Thomas Nixon, Jonathan Heathcote
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * This header file contains a library of advanced C Pre-Processor (CPP) macros
 * which implement various useful functions, such as iteration, in the
 * pre-processor.
 *
 * Though the file name (quite validly) labels this as magic, there should be
 * enough documentation in the comments for a reader only casually familiar
 * with the CPP to be able to understand how everything works.
 *
 * The majority of the magic tricks used in this file are based on those
 * described by pfultz2 in his "Cloak" library:
 *
 * https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
 *
 * Major differences are a greater level of detailed explanation in this
 * implementation and also a refusal to include any macros which require a O(N)
 * macro definitions to handle O(N) arguments (with the exception of DEFERn).
 */

#ifndef CPP_MAGIC_H
#define CPP_MAGIC_H

/**
 * Force the pre-processor to expand the macro a large number of times. Usage:
 *
 *   CPP_MAGIC_EVAL(expression)
 *
 * This is useful when you have a macro which evaluates to a valid macro
 * expression which is not subsequently expanded in the same pass. A contrived,
 * but easy to understand, example of such a macro follows. Note that though
 * this example is contrived, this behaviour is abused to implement bounded
 * recursion in macros such as FOR.
 *
 *   #define A(x) x+1
 *   #define CPP_MAGIC_EMPTY
 *   #define NOT_QUITE_RIGHT(x) A CPP_MAGIC_EMPTY (x)
 *   NOT_QUITE_RIGHT(999)
 *
 * Here's what happens inside the C preprocessor:
 *
 * 1. It sees a macro "NOT_QUITE_RIGHT" and performs a single macro expansion
 *    pass on its arguments. Since the argument is "999" and this isn't a
 * macro,
 *    this is a boring step resulting in no change.
 * 2. The NOT_QUITE_RIGHT macro is substituted for its definition giving "A
 *    CPP_MAGIC_EMPTY() (x)".
 * 3. The expander moves from left-to-right trying to expand the macro:
 *    The first token, A, cannot be expanded since there are no brackets
 *    immediately following it. The second token CPP_MAGIC_EMPTY(), however, can
 * be
 *    expanded (recursively in this manner) and is replaced with "".
 * 4. Expansion continues from the start of the substituted test (which in this
 *    case is just empty), and sees "(999)" but since no macro name is present,
 *    nothing is done. This results in a final expansion of "A (999)".
 *
 * Unfortunately, this doesn't quite meet expectations since you may expect
 * that
 * "A (999)" would have been expanded into "999+1". Unfortunately this requires
 * a second expansion pass but luckily we can force the macro processor to make
 * more passes by abusing the first step of macro expansion: the preprocessor
 * expands arguments in their own pass. If we define a macro which does nothing
 * except produce its arguments e.g.:
 *
 *   #define PASS_THROUGH(...) __VA_ARGS__
 *
 * We can now do "PASS_THROUGH(NOT_QUITE_RIGHT(999))" causing "NOT_QUITE_RIGHT"
 * to be
 * expanded to "A (999)", as described above, when the arguments are expanded.
 * Now when the body of PASS_THROUGH is expanded, "A (999)" gets expanded to
 * "999+1".
 *
 * The CPP_MAGIC_EVAL defined below is essentially equivalent to a large nesting
 * of
 * "PASS_THROUGH(PASS_THROUGH(PASS_THROUGH(..." which results in the
 * preprocessor making a large number of expansion passes over the given
 * expression.
 */
#define CPP_MAGIC_EVAL( ... ) CPP_MAGIC_EVAL1024( __VA_ARGS__ )
#define CPP_MAGIC_EVAL1024( ... ) \
CPP_MAGIC_EVAL512(                \
  CPP_MAGIC_EVAL512(              \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL512( ... ) \
CPP_MAGIC_EVAL256(               \
  CPP_MAGIC_EVAL256(             \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL256( ... ) \
CPP_MAGIC_EVAL128(               \
  CPP_MAGIC_EVAL128(             \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL128( ... ) \
CPP_MAGIC_EVAL64(                \
  CPP_MAGIC_EVAL64(              \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL64( ... ) \
CPP_MAGIC_EVAL32(               \
  CPP_MAGIC_EVAL32(             \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL32( ... ) \
CPP_MAGIC_EVAL16(               \
  CPP_MAGIC_EVAL16(             \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL16( ... ) \
CPP_MAGIC_EVAL8(                \
  CPP_MAGIC_EVAL8(              \
  __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL8( ... ) CPP_MAGIC_EVAL4( CPP_MAGIC_EVAL4( __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL4( ... ) CPP_MAGIC_EVAL2( CPP_MAGIC_EVAL2( __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL2( ... ) CPP_MAGIC_EVAL1( CPP_MAGIC_EVAL1( __VA_ARGS__ ) )
#define CPP_MAGIC_EVAL1( ... ) __VA_ARGS__

/**
 * Macros which expand to common values
 */
#define CPP_MAGIC_PASS( ... ) __VA_ARGS__
#define CPP_MAGIC_EMPTY()
#define CPP_MAGIC_COMMA() ,
#define CPP_MAGIC_SEMICOLON() ;

#define CPP_MAGIC_PLUS() +
#define CPP_MAGIC_ZERO() 0
#define CPP_MAGIC_ONE() 1

/**
 * Causes a function-style macro to require an additional pass to be expanded.
 *
 * This is useful, for example, when trying to implement recursion since the
 * recursive step must not be expanded in a single pass as the pre-processor
 * will catch it and prevent it.
 *
 * Usage:
 *
 *   CPP_MAGIC_DEFER1(IN_NEXT_PASS)(args, to, the, macro)
 *
 * How it works:
 *
 * 1. When CPP_MAGIC_DEFER1 is expanded, first its arguments are expanded which
 * are
 *    simply IN_NEXT_PASS. Since this is a function-style macro and it has no
 *    arguments, nothing will happen.
 * 2. The body of CPP_MAGIC_DEFER1 will now be expanded resulting in
 * CPP_MAGIC_EMPTY() being
 *    deleted. This results in "IN_NEXT_PASS (args, to, the macro)". Note that
 *    since the macro expander has already passed IN_NEXT_PASS by the time it
 *    expands CPP_MAGIC_EMPTY() and so it won't spot that the brackets which
 * remain can
 * be
 *    applied to IN_NEXT_PASS.
 * 3. At this point the macro expansion completes. If one more pass is made,
 *    IN_NEXT_PASS(args, to, the, macro) will be expanded as desired.
 */
#define CPP_MAGIC_DEFER1( id ) id CPP_MAGIC_EMPTY()

/**
 * As with CPP_MAGIC_DEFER1 except here n additional passes are required for
 * DEFERn.
 *
 * The mechanism is analogous.
 *
 * Note that there doesn't appear to be a way of combining DEFERn macros in
 * order to achieve exponentially increasing defers e.g. as is done by
 * CPP_MAGIC_EVAL.
 */
#define CPP_MAGIC_DEFER2( id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )
#define CPP_MAGIC_DEFER3( \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )( )
#define CPP_MAGIC_DEFER4( \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )( )( )
#define CPP_MAGIC_DEFER5(                                                 \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY \
CPP_MAGIC_EMPTY()( )( )( )( )
#define CPP_MAGIC_DEFER6(                                                 \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY \
CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )( )( )( )( )
#define CPP_MAGIC_DEFER7(                                                 \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY \
CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )( )( )( )( )( )
#define CPP_MAGIC_DEFER8(                                                 \
  id ) id CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY \
CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY CPP_MAGIC_EMPTY()( )( )( )( )( )( )( )

/**
 * Indirection around the standard ## concatenation operator. This simply
 * ensures that the arguments are expanded (once) before concatenation.
 */
#define CPP_MAGIC_CAT( a, ... ) a ## __VA_ARGS__
#define CPP_MAGIC_CAT3( a, b, ... ) a ## b ## __VA_ARGS__

/**
 * Get the first argument and ignore the rest.
 */
#define CPP_MAGIC_FIRST( a, ... ) a

/**
 * Get the second argument and ignore the rest.
 */
#define CPP_MAGIC_SECOND( a, b, ... ) b

/**
 * Expects a single input (not containing commas). Returns 1 if the input is
 * CPP_MAGIC_PROBE() and otherwise returns 0.
 *
 * This can be useful as the basis of a CPP_MAGIC_NOT function.
 *
 * This macro abuses the fact that CPP_MAGIC_PROBE() contains a comma while
 * other valid
 * inputs must not.
 */
#define CPP_MAGIC_IS_PROBE( ... ) CPP_MAGIC_SECOND( __VA_ARGS__, 0 )
#define CPP_MAGIC_PROBE() ~, 1

/**
 * Logical negation. 0 is defined as false and everything else as true.
 *
 * When 0, CPP_MAGIC__NOT_0 will be found which evaluates to the
 * CPP_MAGIC_PROBE. When 1 (or any
 * other
 * value) is given, an appropriately named macro won't be found and the
 * concatenated string will be produced. CPP_MAGIC_IS_PROBE then simply checks
 * to see if
 * the CPP_MAGIC_PROBE was returned, cleanly converting the argument into a 1 or
 * 0.
 */
#define CPP_MAGIC_NOT( x ) \
CPP_MAGIC_IS_PROBE(        \
  CPP_MAGIC_CAT(           \
  CPP_MAGIC_NOT_,          \
  x ) )
#define CPP_MAGIC_NOT_0 CPP_MAGIC_PROBE()

/**
 * Macro version of C's famous "cast to bool" operator (i.e. !!) which takes
 * anything and casts it to 0 if it is 0 and 1 otherwise.
 */
#define CPP_MAGIC_BOOL( x ) CPP_MAGIC_NOT( CPP_MAGIC_NOT( x ) )

/**
 * Logical OR. Simply performs a lookup.
 */
#define CPP_MAGIC_OR( a, b ) CPP_MAGIC_CAT3( CPP_MAGIC_OR_, a, b )
#define CPP_MAGIC_OR_00 0
#define CPP_MAGIC_OR_01 1
#define CPP_MAGIC_OR_10 1
#define CPP_MAGIC_OR_11 1

/**
 * Logical AND. Simply performs a lookup.
 */
#define CPP_MAGIC_AND( a, b ) CPP_MAGIC_CAT3( CPP_MAGIC_AND_, a, b )
#define CPP_MAGIC_AND_00 0
#define CPP_MAGIC_AND_01 0
#define CPP_MAGIC_AND_10 0
#define CPP_MAGIC_AND_11 1

/**
 * Macro if statement. Usage:
 *
 *   CPP_MAGIC_IF(c)(expansion when true)
 *
 * Here's how:
 *
 * 1. The preprocessor expands the arguments to _IF casting the condition to
 * '0'
 *    or '1'.
 * 2. The casted condition is concatencated with _IF_ giving _IF_0 or _IF_1.
 * 3. The _IF_0 and _IF_1 macros either returns the argument or doesn't (e.g.
 *    they implement the "choice selection" part of the macro).
 * 4. Note that the "true" clause is in the extra set of brackets; thus these
 *    become the arguments to _IF_0 or _IF_1 and thus a selection is made!
 */
#define CPP_MAGIC_IF( c ) CPP_MAGIC_IMPL_IF( CPP_MAGIC_BOOL( c ) )
#define CPP_MAGIC_IMPL_IF( c ) CPP_MAGIC_CAT( CPP_MAGIC_IMPL_IF_, c )
#define CPP_MAGIC_IMPL_IF_0( ... )
#define CPP_MAGIC_IMPL_IF_1( ... ) __VA_ARGS__

/**
 * Macro if/else statement. Usage:
 *
 *   IF_ELSE(c)( \
 *     expansion when true, \
 *     expansion when false \
 *   )
 *
 * The mechanism is analogous to CPP_MAGIC_IF.
 */
#define CPP_MAGIC_IF_ELSE( c ) CPP_MAGIC_IMPL_IF_ELSE( CPP_MAGIC_BOOL( c ) )
#define CPP_MAGIC_IMPL_IF_ELSE( c ) CPP_MAGIC_CAT( CPP_MAGIC_IMPL_IF_ELSE_, c )
#define CPP_MAGIC_IMPL_IF_ELSE_0( t, f ) f
#define CPP_MAGIC_IMPL_IF_ELSE_1( t, f ) t

/**
 * Macro which checks if it has any arguments. Returns '0' if there are no
 * arguments, '1' otherwise.
 *
 * Limitation: CPP_MAGIC_HAS_ARGS(,1,2,3) returns 0 -- this check essentially
 * only checks
 * that the first argument exists.
 *
 * This macro works as follows:
 *
 * 1. CPP_MAGIC_END_OF_ARGUMENTS_ is concatenated with the first argument.
 * 2. If the first argument is not present then only
 * "CPP_MAGIC_END_OF_ARGUMENTS_" will
 *    remain, otherwise "_END_OF_ARGUMENTS something_here" will remain. This
 *    remaining argument can start with parentheses.
 * 3. In the former case, the CPP_MAGIC_END_OF_ARGUMENTS_(0) macro expands to a
 *    0 when it is expanded. In the latter, a non-zero result remains. If the
 *    first argument started with parentheses these will mostly not contain
 *    only a single 0, but e.g a C cast or some arithmetic operation that will
 *    cause the CPP_MAGIC_BOOL in CPP_MAGIC_END_OF_ARGUMENTS_ to be one.
 * 4. CPP_MAGIC_BOOL is used to force non-zero results into 1 giving the clean 0
 * or 1
 *    output required.
 */
#define CPP_MAGIC_HAS_ARGS( ... ) \
CPP_MAGIC_BOOL(                   \
  CPP_MAGIC_FIRST(                \
  CPP_MAGIC_END_OF_ARGUMENTS_ __VA_ARGS__ )( 0 ) )
#define CPP_MAGIC_END_OF_ARGUMENTS_( ... ) \
CPP_MAGIC_BOOL(                            \
  CPP_MAGIC_FIRST(                         \
  __VA_ARGS__ ) )

/**
 * Macro map/list comprehension. Usage:
 *
 *   CPP_MAGIC_MAP(op, sep, ...)
 *
 * Produces a 'sep()'-separated list of the result of op(arg) for each arg.
 *
 * Example Usage:
 *
 *   #define MAKE_HAPPY(x) happy_##x
 *   #define CPP_MAGIC_COMMA() ,
 *   CPP_MAGIC_MAP(MAKE_HAPPY, CPP_MAGIC_COMMA, 1,2,3)
 *
 * Which expands to:
 *
 *    happy_1 , happy_2 , happy_3
 *
 * How it works:
 *
 * 1. The CPP_MAGIC_MAP macro simply maps the inner CPP_MAGIC_MAP_INNER function
 * in an CPP_MAGIC_EVAL which
 *    forces it to be expanded a large number of times, thus enabling many
 * steps
 *    of iteration (see step 6).
 * 2. The CPP_MAGIC_MAP_INNER macro is substituted for its body.
 * 3. In the body, op(cur_val) is substituted giving the value for this
 *    iteration.
 * 4. The CPP_MAGIC_IF macro expands according to whether further iterations are
 * required.
 *    This expansion either produces _IF_0 or _IF_1.
 * 5. Since the CPP_MAGIC_IF is followed by a set of brackets containing the "if
 * true"
 *    clause, these become the argument to _IF_0 or _IF_1. At this point, the
 *    macro in the brackets will be expanded giving the separator followed by
 *    _MAP_INNER CPP_MAGIC_EMPTY()()(op, sep, __VA_ARGS__).
 * 5. If the CPP_MAGIC_IF was not taken, the above will simply be discarded and
 * everything
 *    stops. If the CPP_MAGIC_IF is taken, The expression is then processed a
 * second time
 *    yielding "_MAP_INNER()(op, sep, __VA_ARGS__)". Note that this call looks
 *    very similar to the  essentially the same as the original call except the
 *    first argument has been dropped.
 * 6. At this point expansion of CPP_MAGIC_MAP_INNER will terminate. However,
 * since we
 * can
 *    force more rounds of expansion using CPP_MAGIC_EVAL1. In the
 * argument-expansion
 * pass
 *    of the CPP_MAGIC_EVAL1, _MAP_INNER() is expanded to CPP_MAGIC_MAP_INNER
 * which is then
 * expanded
 *    using the arguments which follow it as in step 2-5. This is followed by a
 *    second expansion pass as the substitution of CPP_MAGIC_EVAL1() is expanded
 * executing
 *    2-5 a second time. This results in up to two iterations occurring. Using
 *    many nested CPP_MAGIC_EVAL1 macros, i.e. the very-deeply-nested
 * CPP_MAGIC_EVAL macro, will in
 *    this manner produce further iterations, hence the outer CPP_MAGIC_MAP
 * macro doing
 *    this for us.
 *
 * Important tricks used:
 *
 * * If we directly produce "CPP_MAGIC_MAP_INNER" in an expansion of
 * CPP_MAGIC_MAP_INNER, a special
 *   case in the preprocessor will prevent it being expanded in the future,
 * even
 *   if we CPP_MAGIC_EVAL.  As a result, the CPP_MAGIC_MAP_INNER macro carefully
 * only expands to
 *   something containing "_MAP_INNER()" which requires a further expansion
 * step
 *   to invoke CPP_MAGIC_MAP_INNER and thus implementing the recursion.
 * * To prevent _MAP_INNER being expanded within the macro we must first defer
 * its
 *   expansion during its initial pass as an argument to _IF_0 or _IF_1. We
 * must
 *   then defer its expansion a second time as part of the body of the _IF_0.
 * As
 *   a result hence the CPP_MAGIC_DEFER2.
 * * _MAP_INNER seemingly gets away with producing itself because it actually
 * only
 *   produces CPP_MAGIC_MAP_INNER. It just happens that when _MAP_INNER() is
 * expanded in
 *   this case it is followed by some arguments which get consumed by
 * CPP_MAGIC_MAP_INNER
 *   and produce a _MAP_INNER.  As such, the macro expander never marks
 *   _MAP_INNER as expanding to itself and thus it will still be expanded in
 *   future productions of itself.
 */
#define CPP_MAGIC_MAP( op, sep, ... )                              \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )( CPP_MAGIC_EVAL( \
  CPP_MAGIC_MAP_INNER( op, sep, __VA_ARGS__ ) ) )
#define CPP_MAGIC_MAP_INNER( op, sep, cur_val, ... )                            \
op( cur_val )                                                                   \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                              \
  sep() CPP_MAGIC_DEFER2( CPP_MAGIC_IMPL_MAP_INNER )( )( op, sep, __VA_ARGS__ ) \
)
#define CPP_MAGIC_IMPL_MAP_INNER() CPP_MAGIC_MAP_INNER

/**
 * This is a variant of the CPP_MAGIC_MAP macro which also includes as an
 * argument to the
 * operation a valid C variable name which is different for each iteration.
 *
 * Usage:
 *   CPP_MAGIC_MAP_WITH_ID(op, sep, ...)
 *
 * Where op is a macro op(val, id) which takes a list value and an ID. This ID
 * will simply be a unary number using the digit "I", that is, I, II, III,
 * IIII,
 * and so on.
 *
 * Example:
 *
 *   #define MAKE_STATIC_VAR(type, name) static type name;
 *   CPP_MAGIC_MAP_WITH_ID(MAKE_STATIC_VAR, CPP_MAGIC_EMPTY, int, int, int,
 * bool, char)
 *
 * Which expands to:
 *
 *   static int I; static int II; static int III; static bool IIII; static char
 * IIIII;
 *
 * The mechanism is analogous to the CPP_MAGIC_MAP macro.
 */
#define CPP_MAGIC_MAP_WITH_ID( op, sep, ... )                      \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )( CPP_MAGIC_EVAL( \
  CPP_MAGIC_MAP_WITH_ID_INNER(                                     \
  op, sep, I,                                                      \
  __VA_ARGS__ ) ) )
#define CPP_MAGIC_MAP_WITH_ID_INNER( op, sep, id, cur_val, ... )                \
op( cur_val, id )                                                               \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                              \
  sep() CPP_MAGIC_DEFER2( CPP_MAGIC_IMPL_MAP_WITH_ID_INNER )( )( op, sep,       \
                                                                 CPP_MAGIC_CAT( \
  id, I ), __VA_ARGS__ )                                                        \
)
#define CPP_MAGIC_IMPL_MAP_WITH_ID_INNER() CPP_MAGIC_MAP_WITH_ID_INNER

/**
 * This is a variant of the CPP_MAGIC_MAP macro which iterates over pairs rather
 * than
 * singletons.
 *
 * Usage:
 *   CPP_MAGIC_MAP_PAIRS(op, sep, ...)
 *
 * Where op is a macro op(val_1, val_2) which takes two list values.
 *
 * Example:
 *
 *   #define MAKE_STATIC_VAR(type, name) static type name;
 *   CPP_MAGIC_MAP_PAIRS(MAKE_STATIC_VAR, CPP_MAGIC_EMPTY, char, my_char, int,
 * my_int)
 *
 * Which expands to:
 *
 *   static char my_char; static int my_int;
 *
 * The mechanism is analogous to the CPP_MAGIC_MAP macro.
 */
#define CPP_MAGIC_MAP_PAIRS( op, sep, ... )                        \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )( CPP_MAGIC_EVAL( \
  CPP_MAGIC_MAP_PAIRS_INNER(                                       \
  op, sep,                                                         \
  __VA_ARGS__ ) ) )
#define CPP_MAGIC_MAP_PAIRS_INNER( op, sep, cur_val_1, cur_val_2, ... )      \
op( cur_val_1, cur_val_2 )                                                   \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                           \
  sep() CPP_MAGIC_DEFER2( CPP_MAGIC_IMPL_MAP_PAIRS_INNER )( )( op, sep,      \
                                                               __VA_ARGS__ ) \
)
#define CPP_MAGIC_IMPL_MAP_PAIRS_INNER() CPP_MAGIC_MAP_PAIRS_INNER

/**
 * This is a variant of the CPP_MAGIC_MAP macro which iterates over a
 * two-element sliding
 * window.
 *
 * Usage:
 *   CPP_MAGIC_MAP_SLIDE(op, last_op, sep, ...)
 *
 * Where op is a macro op(val_1, val_2) which takes the two list values
 * currently in the window. last_op is a macro taking a single value which is
 * called for the last argument.
 *
 * Example:
 *
 *   #define SIMON_SAYS_OP(simon, next)
 * CPP_MAGIC_IF(CPP_MAGIC_NOT(simon()))(next)
 *   #define SIMON_SAYS_LAST_OP(val) last_but_not_least_##val
 *   #define SIMON_SAYS() 0
 *
 *   CPP_MAGIC_MAP_SLIDE(SIMON_SAYS_OP, SIMON_SAYS_LAST_OP, CPP_MAGIC_EMPTY,
 * wiggle, SIMON_SAYS,
 * dance, move, SIMON_SAYS, boogie, stop)
 *
 * Which expands to:
 *
 *   dance boogie last_but_not_least_stop
 *
 * The mechanism is analogous to the CPP_MAGIC_MAP macro.
 */
#define CPP_MAGIC_MAP_SLIDE( op, last_op, sep, ... )               \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )( CPP_MAGIC_EVAL( \
  CPP_MAGIC_MAP_SLIDE_INNER(                                       \
  op, last_op, sep,                                                \
  __VA_ARGS__ ) ) )
#define CPP_MAGIC_MAP_SLIDE_INNER( op, last_op, sep, cur_val, ... )              \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )( op(                           \
  cur_val,                                                                       \
  CPP_MAGIC_FIRST( __VA_ARGS__ ) ) )                                             \
CPP_MAGIC_IF( CPP_MAGIC_NOT( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) ) )( last_op(     \
  cur_val ) )                                                                    \
CPP_MAGIC_IF( CPP_MAGIC_HAS_ARGS( __VA_ARGS__ ) )(                               \
  sep() CPP_MAGIC_DEFER2( CPP_MAGIC_IMPL_MAP_SLIDE_INNER )( )( op, last_op, sep, \
                                                               __VA_ARGS__ )     \
)
#define CPP_MAGIC_IMPL_MAP_SLIDE_INNER() CPP_MAGIC_MAP_SLIDE_INNER

/**
 * Strip any excess commas from a set of arguments.
 */
#define CPP_MAGIC_REMOVE_TRAILING_COMMAS( ... ) \
CPP_MAGIC_MAP( CPP_MAGIC_PASS, CPP_MAGIC_COMMA, __VA_ARGS__ )

#endif
