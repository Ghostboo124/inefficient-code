"""
A Really inefficient way of putting together a sentence
Error Codes:
    0: Everything worked!
    1: You entered a phrase that was more than 32 characters long, I will be adding support for more than 4 threads in the future.
    2: Your CPU has less than 4 threads, please upgrade your CPU or figure out some other way to run this script.
    3: Errors occured during imports, use the advice printed to the screen at time of error.
"""

# Imports
try:
    from time import time
    import os
    from os import urandom as _urandom
    from time import monotonic as _time
    from _weakref import ref
    from itertools import count as _count
    import _thread
    import sys as _sys
    from collections import deque as _deque
except ImportError:
    print("Uhh, all these modules should be included in the python executable, if not then use:")
    print("    pip install time os itertools sys collections")
    print("If there are any errors, then remove the package that is causing the error")
    print("Exiting with Code: 3")
    exit(3)

# Threads
try:
    from threadCode import thread0
    from threadCode import thread1
    from threadCode import thread2
    from threadCode import thread3
except:
    print("The files in the threadCode directory failed to import, plese make sure that they haven't been removed and contain the files: \n\tthread0.py\n\tthread1.py\n\tthread2.py\n\tthread3.py")
    print("If any of these files are missing, please use `git pull` to pull them or if that doesn't fix it, then download them from the github page")

# MY CODE

if os.cpu_count() <= 4:
    print("You need more than 4 threads!")
    print("Exiting with Code: 2")
    exit(2)
elif os.cpu_count() >= 32:
    print("What CPU are you running, that is more than 32 threads! This is Definitly Supported")
else:
    print("Your CPU is supported")

sentenceUser = input("What is the sentence you want to make (no more than 32 chars): ").lower()
sentenceUs = ""
start = time()

thread0Return = ""
que0 = []

thread1Return = ""
que1 = []

thread2Return = ""
que2 = []

thread3Return = ""
que3 = []

counter = 1
for letter in sentenceUser:
    if counter == 1 or counter == 5 or counter == 9 or counter == 13 or counter == 17 or counter == 21 or counter == 25 or counter == 29:
        que0.append(letter)
    if counter == 2 or counter == 6 or counter == 10 or counter == 14 or counter == 18 or counter == 22 or counter == 26 or counter == 30:
        que1.append(letter)
    if counter == 3 or counter == 7 or counter == 11 or counter == 15 or counter == 19 or counter == 23 or counter == 27 or counter == 31:
        que2.append(letter)
    if counter == 4 or counter == 8 or counter == 12 or counter == 16 or counter == 20 or counter == 24 or counter == 38 or counter == 32:
        que3.append(letter)
    # else:
    #     que0.append(letter)
    counter += 1
    if counter > 32:
        print("Hey, you weren't supposed to make your phrase more than 32 characters!, we are leaving now")
        print("Exiting with Code: 1")
        exit(1)

# NOT MY CODE
__all__ = ['get_ident', 'active_count', 'Condition', 'current_thread',
           'enumerate', 'main_thread', 'TIMEOUT_MAX',
           'Event', 'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Thread',
           'Barrier', 'BrokenBarrierError', 'Timer', 'ThreadError',
           'setprofile', 'settrace', 'local', 'stack_size',
           'excepthook', 'ExceptHookArgs', 'gettrace', 'getprofile',
           'setprofile_all_threads','settrace_all_threads', 'WeakSet']
_active = {}    # maps thread id to Thread object
_limbo = {}

# Rename some stuff so "from threading import *" is safe
_start_new_thread = _thread.start_new_thread
_daemon_threads_allowed = _thread.daemon_threads_allowed
_allocate_lock = _thread.allocate_lock
_set_sentinel = _thread._set_sentinel
get_ident = _thread.get_ident
_shutdown_locks_lock = _allocate_lock()
_shutdown_locks = set()
_profile_hook = None
_trace_hook = None
try:
    _is_main_interpreter = _thread._is_main_interpreter
except AttributeError:
    # See https://github.com/python/cpython/issues/112826.
    # We can pretend a subinterpreter is the main interpreter for the
    # sake of _shutdown(), since that only means we do not wait for the
    # subinterpreter's threads to finish.  Instead, they will be stopped
    # later by the mechanism we use for daemon threads.  The likelihood
    # of this case is small because rarely will the _thread module be
    # replaced by a module without _is_main_interpreter().
    # Furthermore, this is all irrelevant in applications
    # that do not use subinterpreters.
    def _is_main_interpreter():
        return True
try:
    get_native_id = _thread.get_native_id
    _HAVE_THREAD_NATIVE_ID = True
    __all__.append('get_native_id')
except AttributeError:
    _HAVE_THREAD_NATIVE_ID = False
ThreadError = _thread.error
try:
    _CRLock = _thread.RLock
except AttributeError:
    _CRLock = None
TIMEOUT_MAX = _thread.TIMEOUT_MAX
del _thread

try:
    from _thread import (_excepthook as excepthook,
                         _ExceptHookArgs as ExceptHookArgs)
except ImportError:
    # Simple Python implementation if _thread._excepthook() is not available
    # traceback.py

    from collections import (abc as _abc,
                             namedtuple as _namedtuple)
    from itertools import islice as _islice
    from linecache import lazycache as _lazycache
    from linecache import checkcache as _checkcache
    from linecache import getline as _getline
    from contextlib import suppress as _suppress
    import textwrap as _textwrap

    class _Sentinel:
        def __repr__(self):
            return "<implicit>"
    _sentinel = _Sentinel()
    _cause_message = (
        "\nThe above exception was the direct cause "
        "of the following exception:\n\n")

    _context_message = (
        "\nDuring handling of the above exception, "
        "another exception occurred:\n\n")
    sys = _sys
    _WIDE_CHAR_SPECIFIERS = "WF"
    _RECURSIVE_CUTOFF = 3
    _MAX_CANDIDATE_ITEMS = 750
    _MAX_STRING_SIZE = 40
    _MOVE_COST = 2
    _CASE_COST = 1

    def _parse_value_tb(exc, value, tb):
        if (value is _sentinel) != (tb is _sentinel):
            raise ValueError("Both or neither of value and tb must be given")
        if value is tb is _sentinel:
            if exc is not None:
                if isinstance(exc, BaseException):
                    return exc, exc.__traceback__

                raise TypeError(f'Exception expected for value, '
                                f'{type(exc).__name__} found')
            else:
                return None, None
        return value, tb

    class _ExceptionPrintContext:
        def __init__(self):
            self.seen = set()
            self.exception_group_depth = 0
            self.need_close = False

    def indent(self):
        return ' ' * (2 * self.exception_group_depth)

    def emit(self, text_gen, margin_char=None):
        if margin_char is None:
            margin_char = '|'
        indent_str = self.indent()
        if self.exception_group_depth:
            indent_str += margin_char + ' '

        if isinstance(text_gen, str):
            yield _textwrap.indent(text_gen, indent_str, lambda line: True)
        else:
            for text in text_gen:
                yield _textwrap.indent(text, indent_str, lambda line: True)

    def _format_final_exc_line(etype, value):
        valuestr = _safe_string(value, 'exception')
        if value is None or not valuestr:
            line = "%s\n" % etype
        else:
            line = "%s: %s\n" % (etype, valuestr)
        return line

    def _substitution_cost(ch_a, ch_b):
        if ch_a == ch_b:
            return 0
        if ch_a.lower() == ch_b.lower():
            return _CASE_COST
        return _MOVE_COST

    def _levenshtein_distance(a, b, max_cost):
        # A Python implementation of Python/suggestions.c:levenshtein_distance.

        # Both strings are the same
        if a == b:
            return 0

        # Trim away common affixes
        pre = 0
        while a[pre:] and b[pre:] and a[pre] == b[pre]:
            pre += 1
        a = a[pre:]
        b = b[pre:]
        post = 0
        while a[:post or None] and b[:post or None] and a[post-1] == b[post-1]:
            post -= 1
        a = a[:post or None]
        b = b[:post or None]
        if not a or not b:
            return _MOVE_COST * (len(a) + len(b))
        if len(a) > _MAX_STRING_SIZE or len(b) > _MAX_STRING_SIZE:
            return max_cost + 1

        # Prefer shorter buffer
        if len(b) < len(a):
            a, b = b, a

        # Quick fail when a match is impossible
        if (len(b) - len(a)) * _MOVE_COST > max_cost:
            return max_cost + 1

        # Instead of producing the whole traditional len(a)-by-len(b)
        # matrix, we can update just one row in place.
        # Initialize the buffer row
        row = list(range(_MOVE_COST, _MOVE_COST * (len(a) + 1), _MOVE_COST))

        result = 0
        for bindex in range(len(b)):
            bchar = b[bindex]
            distance = result = bindex * _MOVE_COST
            minimum = sys.maxsize
            for index in range(len(a)):
                # 1) Previous distance in this row is cost(b[:b_index], a[:index])
                substitute = distance + _substitution_cost(bchar, a[index])
                # 2) cost(b[:b_index], a[:index+1]) from previous row
                distance = row[index]
                # 3) existing result is cost(b[:b_index+1], a[index])

                insert_delete = min(result, distance) + _MOVE_COST
                result = min(insert_delete, substitute)

                # cost(b[:b_index+1], a[:index+1])
                row[index] = result
                if result < minimum:
                    minimum = result
            if minimum > max_cost:
                # Everything in this row is too big, so bail early.
                return max_cost + 1
        return result

    def _byte_offset_to_character_offset(str, offset):
        as_utf8 = str.encode('utf-8')
        return len(as_utf8[:offset].decode("utf-8", errors="replace"))


    _Anchors = _namedtuple(
        "_Anchors",
        [
            "left_end_offset",
            "right_start_offset",
            "primary_char",
            "secondary_char",
        ],
        defaults=["~", "^"]
    )

    def _get_code_position(code, instruction_index):
        if instruction_index < 0:
            return (None, None, None, None)
        positions_gen = code.co_positions()
        return next(_islice(positions_gen, instruction_index // 2, None))

    def _extract_caret_anchors_from_line_segment(segment):
        import _ast

        try:
            tree = _ast.parse(segment)
        except SyntaxError:
            return None

        if len(tree.body) != 1:
            return None

        normalize = lambda offset: _byte_offset_to_character_offset(segment, offset)
        statement = tree.body[0]
        match statement:
            case _ast.Expr(expr):
                match expr:
                    case _ast.BinOp():
                        operator_start = normalize(expr.left.end_col_offset)
                        operator_end = normalize(expr.right.col_offset)
                        operator_str = segment[operator_start:operator_end]
                        operator_offset = len(operator_str) - len(operator_str.lstrip())

                        left_anchor = expr.left.end_col_offset + operator_offset
                        right_anchor = left_anchor + 1
                        if (
                            operator_offset + 1 < len(operator_str)
                            and not operator_str[operator_offset + 1].isspace()
                        ):
                            right_anchor += 1

                        while left_anchor < len(segment) and ((ch := segment[left_anchor]).isspace() or ch in ")#"):
                            left_anchor += 1
                            right_anchor += 1
                        return _Anchors(normalize(left_anchor), normalize(right_anchor))
                    case _ast.Subscript():
                        left_anchor = normalize(expr.value.end_col_offset)
                        right_anchor = normalize(expr.slice.end_col_offset + 1)
                        while left_anchor < len(segment) and ((ch := segment[left_anchor]).isspace() or ch != "["):
                            left_anchor += 1
                        while right_anchor < len(segment) and ((ch := segment[right_anchor]).isspace() or ch != "]"):
                            right_anchor += 1
                        if right_anchor < len(segment):
                            right_anchor += 1
                        return _Anchors(left_anchor, right_anchor)

        return None

    def _display_width(line, offset):
        """Calculate the extra amount of width space the given source
        code segment might take if it were to be displayed on a fixed
        width output device. Supports wide unicode characters and emojis."""

        # Fast track for ASCII-only strings
        if line.isascii():
            return offset

        import unicodedata

        return sum(
            2 if unicodedata.east_asian_width(char) in _WIDE_CHAR_SPECIFIERS else 1
            for char in line[:offset]
        )

    class FrameSummary:
        """Information about a single frame from a traceback.

        - :attr:`filename` The filename for the frame.
        - :attr:`lineno` The line within filename for the frame that was
        active when the frame was captured.
        - :attr:`name` The name of the function or method that was executing
        when the frame was captured.
        - :attr:`line` The text from the linecache module for the
        of code that was running when the frame was captured.
        - :attr:`locals` Either None if locals were not supplied, or a dict
        mapping the name to the repr() of the variable.
        """

        __slots__ = ('filename', 'lineno', 'end_lineno', 'colno', 'end_colno',
                    'name', '_line', 'locals')

        def __init__(self, filename, lineno, name, *, lookup_line=True,
                locals=None, line=None,
                end_lineno=None, colno=None, end_colno=None):
            """Construct a FrameSummary.

            :param lookup_line: If True, `linecache` is consulted for the source
                code line. Otherwise, the line will be looked up when first needed.
            :param locals: If supplied the frame locals, which will be captured as
                object representations.
            :param line: If provided, use this instead of looking up the line in
                the linecache.
            """
            self.filename = filename
            self.lineno = lineno
            self.name = name
            self._line = line
            if lookup_line:
                self.line
            self.locals = {k: _safe_string(v, 'local', func=repr)
                for k, v in locals.items()} if locals else None
            self.end_lineno = end_lineno
            self.colno = colno
            self.end_colno = end_colno

        def __eq__(self, other):
            if isinstance(other, FrameSummary):
                return (self.filename == other.filename and
                        self.lineno == other.lineno and
                        self.name == other.name and
                        self.locals == other.locals)
            if isinstance(other, tuple):
                return (self.filename, self.lineno, self.name, self.line) == other
            return NotImplemented

        def __getitem__(self, pos):
            return (self.filename, self.lineno, self.name, self.line)[pos]

        def __iter__(self):
            return iter([self.filename, self.lineno, self.name, self.line])

        def __repr__(self):
            return "<FrameSummary file {filename}, line {lineno} in {name}>".format(
                filename=self.filename, lineno=self.lineno, name=self.name)

        def __len__(self):
            return 4

        @property
        def _original_line(self):
            # Returns the line as-is from the source, without modifying whitespace.
            self.line
            return self._line

        @property
        def line(self):
            if self._line is None:
                if self.lineno is None:
                    return None
                self._line = _getline(self.filename, self.lineno)
            return self._line.strip()

    class StackSummary(list):
        """A list of FrameSummary objects, representing a stack of frames."""

        @classmethod
        def extract(klass, frame_gen, *, limit=None, lookup_lines=True,
                capture_locals=False):
            """Create a StackSummary from a traceback or stack object.

            :param frame_gen: A generator that yields (frame, lineno) tuples
                whose summaries are to be included in the stack.
            :param limit: None to include all frames or the number of frames to
                include.
            :param lookup_lines: If True, lookup lines for each frame immediately,
                otherwise lookup is deferred until the frame is rendered.
            :param capture_locals: If True, the local variables from each frame will
                be captured as object representations into the FrameSummary.
            """
            def extended_frame_gen():
                for f, lineno in frame_gen:
                    yield f, (lineno, None, None, None)

            return klass._extract_from_extended_frame_gen(
                extended_frame_gen(), limit=limit, lookup_lines=lookup_lines,
                capture_locals=capture_locals)

        @classmethod
        def _extract_from_extended_frame_gen(klass, frame_gen, *, limit=None,
                lookup_lines=True, capture_locals=False):
            # Same as extract but operates on a frame generator that yields
            # (frame, (lineno, end_lineno, colno, end_colno)) in the stack.
            # Only lineno is required, the remaining fields can be None if the
            # information is not available.
            if limit is None:
                limit = getattr(sys, 'tracebacklimit', None)
                if limit is not None and limit < 0:
                    limit = 0
            if limit is not None:
                if limit >= 0:
                    frame_gen = _islice(frame_gen, limit)
                else:
                    frame_gen = _deque(frame_gen, maxlen=-limit)

            result = klass()
            fnames = set()
            for f, (lineno, end_lineno, colno, end_colno) in frame_gen:
                co = f.f_code
                filename = co.co_filename
                name = co.co_name

                fnames.add(filename)
                _lazycache(filename, f.f_globals)
                # Must defer line lookups until we have called checkcache.
                if capture_locals:
                    f_locals = f.f_locals
                else:
                    f_locals = None
                result.append(FrameSummary(
                    filename, lineno, name, lookup_line=False, locals=f_locals,
                    end_lineno=end_lineno, colno=colno, end_colno=end_colno))
            for filename in fnames:
                _checkcache(filename)
            # If immediate lookup was desired, trigger lookups now.
            if lookup_lines:
                for f in result:
                    f.line
            return result

        @classmethod
        def from_list(klass, a_list):
            """
            Create a StackSummary object from a supplied list of
            FrameSummary objects or old-style list of tuples.
            """
            # While doing a fast-path check for isinstance(a_list, StackSummary) is
            # appealing, idlelib.run.cleanup_traceback and other similar code may
            # break this by making arbitrary frames plain tuples, so we need to
            # check on a frame by frame basis.
            result = StackSummary()
            for frame in a_list:
                if isinstance(frame, FrameSummary):
                    result.append(frame)
                else:
                    filename, lineno, name, line = frame
                    result.append(FrameSummary(filename, lineno, name, line=line))
            return result

        def format_frame_summary(self, frame_summary):
            """Format the lines for a single FrameSummary.

            Returns a string representing one frame involved in the stack. This
            gets called for every frame to be printed in the stack summary.
            """
            row = []
            row.append('  File "{}", line {}, in {}\n'.format(
                frame_summary.filename, frame_summary.lineno, frame_summary.name))
            if frame_summary.line:
                stripped_line = frame_summary.line.strip()
                row.append('    {}\n'.format(stripped_line))

                line = frame_summary._original_line
                orig_line_len = len(line)
                frame_line_len = len(frame_summary.line.lstrip())
                stripped_characters = orig_line_len - frame_line_len
                if (
                    frame_summary.colno is not None
                    and frame_summary.end_colno is not None
                ):
                    start_offset = _byte_offset_to_character_offset(
                        line, frame_summary.colno)
                    end_offset = _byte_offset_to_character_offset(
                        line, frame_summary.end_colno)
                    code_segment = line[start_offset:end_offset]

                    anchors = None
                    if frame_summary.lineno == frame_summary.end_lineno:
                        with _suppress(Exception):
                            anchors = _extract_caret_anchors_from_line_segment(code_segment)
                    else:
                        # Don't count the newline since the anchors only need to
                        # go up until the last character of the line.
                        end_offset = len(line.rstrip())

                    # show indicators if primary char doesn't span the frame line
                    if end_offset - start_offset < len(stripped_line) or (
                            anchors and anchors.right_start_offset - anchors.left_end_offset > 0):
                        # When showing this on a terminal, some of the non-ASCII characters
                        # might be rendered as double-width characters, so we need to take
                        # that into account when calculating the length of the line.
                        dp_start_offset = _display_width(line, start_offset) + 1
                        dp_end_offset = _display_width(line, end_offset) + 1

                        row.append('    ')
                        row.append(' ' * (dp_start_offset - stripped_characters))

                        if anchors:
                            dp_left_end_offset = _display_width(code_segment, anchors.left_end_offset)
                            dp_right_start_offset = _display_width(code_segment, anchors.right_start_offset)
                            row.append(anchors.primary_char * dp_left_end_offset)
                            row.append(anchors.secondary_char * (dp_right_start_offset - dp_left_end_offset))
                            row.append(anchors.primary_char * (dp_end_offset - dp_start_offset - dp_right_start_offset))
                        else:
                            row.append('^' * (dp_end_offset - dp_start_offset))

                        row.append('\n')

            if frame_summary.locals:
                for name, value in sorted(frame_summary.locals.items()):
                    row.append('    {name} = {value}\n'.format(name=name, value=value))

            return ''.join(row)

        def format(self):
            """Format the stack ready for printing.

            Returns a list of strings ready for printing.  Each string in the
            resulting list corresponds to a single frame from the stack.
            Each string ends in a newline; the strings may contain internal
            newlines as well, for those items with source text lines.

            For long sequences of the same frame and line, the first few
            repetitions are shown, followed by a summary line stating the exact
            number of further repetitions.
            """
            result = []
            last_file = None
            last_line = None
            last_name = None
            count = 0
            for frame_summary in self:
                formatted_frame = self.format_frame_summary(frame_summary)
                if formatted_frame is None:
                    continue
                if (last_file is None or last_file != frame_summary.filename or
                    last_line is None or last_line != frame_summary.lineno or
                    last_name is None or last_name != frame_summary.name):
                    if count > _RECURSIVE_CUTOFF:
                        count -= _RECURSIVE_CUTOFF
                        result.append(
                            f'  [Previous line repeated {count} more '
                            f'time{"s" if count > 1 else ""}]\n'
                        )
                    last_file = frame_summary.filename
                    last_line = frame_summary.lineno
                    last_name = frame_summary.name
                    count = 0
                count += 1
                if count > _RECURSIVE_CUTOFF:
                    continue
                result.append(formatted_frame)

            if count > _RECURSIVE_CUTOFF:
                count -= _RECURSIVE_CUTOFF
                result.append(
                    f'  [Previous line repeated {count} more '
                    f'time{"s" if count > 1 else ""}]\n'
                )
            return result
    
    def _walk_tb_with_full_positions(tb):
        # Internal version of walk_tb that yields full code positions including
        # end line and column information.
        while tb is not None:
            positions = _get_code_position(tb.tb_frame.f_code, tb.tb_lasti)
            # Yield tb_lineno when co_positions does not have a line number to
            # maintain behavior with walk_tb.
            if positions[0] is None:
                yield tb.tb_frame, (tb.tb_lineno, ) + positions[1:]
            else:
                yield tb.tb_frame, positions
            tb = tb.tb_next
    
    def _compute_suggestion_error(exc_value, tb, wrong_name):
        if wrong_name is None or not isinstance(wrong_name, str):
            return None
        if isinstance(exc_value, AttributeError):
            obj = exc_value.obj
            try:
                d = dir(obj)
            except Exception:
                return None
        elif isinstance(exc_value, ImportError):
            try:
                mod = __import__(exc_value.name)
                d = dir(mod)
            except Exception:
                return None
        else:
            assert isinstance(exc_value, NameError)
            # find most recent frame
            if tb is None:
                return None
            while tb.tb_next is not None:
                tb = tb.tb_next
            frame = tb.tb_frame
            d = (
                list(frame.f_locals)
                + list(frame.f_globals)
                + list(frame.f_builtins)
            )

            # Check first if we are in a method and the instance
            # has the wrong name as attribute
            if 'self' in frame.f_locals:
                self = frame.f_locals['self']
                if hasattr(self, wrong_name):
                    return f"self.{wrong_name}"

        # Compute closest match

        if len(d) > _MAX_CANDIDATE_ITEMS:
            return None
        wrong_name_len = len(wrong_name)
        if wrong_name_len > _MAX_STRING_SIZE:
            return None
        best_distance = wrong_name_len
        suggestion = None
        for possible_name in d:
            if possible_name == wrong_name:
                # A missing attribute is "found". Don't suggest it (see GH-88821).
                continue
            # No more than 1/3 of the involved characters should need changed.
            max_distance = (len(possible_name) + wrong_name_len + 3) * _MOVE_COST // 6
            # Don't take matches we've already beaten.
            max_distance = min(max_distance, best_distance - 1)
            current_distance = _levenshtein_distance(wrong_name, possible_name, max_distance)
            if current_distance > max_distance:
                continue
            if not suggestion or current_distance < best_distance:
                suggestion = possible_name
                best_distance = current_distance
        return suggestion

    def _safe_string(value, what, func=str):
        try:
            return func(value)
        except:
            return f'<{what} {func.__name__}() failed>'

    class TracebackException:
        """An exception ready for rendering.

        The traceback module captures enough attributes from the original exception
        to this intermediary form to ensure that no references are held, while
        still being able to fully print or format it.

        max_group_width and max_group_depth control the formatting of exception
        groups. The depth refers to the nesting level of the group, and the width
        refers to the size of a single exception group's exceptions array. The
        formatted output is truncated when either limit is exceeded.

        Use `from_exception` to create TracebackException instances from exception
        objects, or the constructor to create TracebackException instances from
        individual components.

        - :attr:`__cause__` A TracebackException of the original *__cause__*.
        - :attr:`__context__` A TracebackException of the original *__context__*.
        - :attr:`exceptions` For exception groups - a list of TracebackException
        instances for the nested *exceptions*.  ``None`` for other exceptions.
        - :attr:`__suppress_context__` The *__suppress_context__* value from the
        original exception.
        - :attr:`stack` A `StackSummary` representing the traceback.
        - :attr:`exc_type` The class of the original traceback.
        - :attr:`filename` For syntax errors - the filename where the error
        occurred.
        - :attr:`lineno` For syntax errors - the linenumber where the error
        occurred.
        - :attr:`end_lineno` For syntax errors - the end linenumber where the error
        occurred. Can be `None` if not present.
        - :attr:`text` For syntax errors - the text where the error
        occurred.
        - :attr:`offset` For syntax errors - the offset into the text where the
        error occurred.
        - :attr:`end_offset` For syntax errors - the end offset into the text where
        the error occurred. Can be `None` if not present.
        - :attr:`msg` For syntax errors - the compiler error message.
        """

        def __init__(self, exc_type, exc_value, exc_traceback, *, limit=None,
                lookup_lines=True, capture_locals=False, compact=False,
                max_group_width=15, max_group_depth=10, _seen=None):
            # NB: we need to accept exc_traceback, exc_value, exc_traceback to
            # permit backwards compat with the existing API, otherwise we
            # need stub thunk objects just to glue it together.
            # Handle loops in __cause__ or __context__.
            is_recursive_call = _seen is not None
            if _seen is None:
                _seen = set()
            _seen.add(id(exc_value))

            self.max_group_width = max_group_width
            self.max_group_depth = max_group_depth

            self.stack = StackSummary._extract_from_extended_frame_gen(
                _walk_tb_with_full_positions(exc_traceback),
                limit=limit, lookup_lines=lookup_lines,
                capture_locals=capture_locals)
            self.exc_type = exc_type
            # Capture now to permit freeing resources: only complication is in the
            # unofficial API _format_final_exc_line
            self._str = _safe_string(exc_value, 'exception')
            try:
                self.__notes__ = getattr(exc_value, '__notes__', None)
            except Exception as e:
                self.__notes__ = [
                    f'Ignored error getting __notes__: {_safe_string(e, '__notes__', repr)}']

            if exc_type and issubclass(exc_type, SyntaxError):
                # Handle SyntaxError's specially
                self.filename = exc_value.filename
                lno = exc_value.lineno
                self.lineno = str(lno) if lno is not None else None
                end_lno = exc_value.end_lineno
                self.end_lineno = str(end_lno) if end_lno is not None else None
                self.text = exc_value.text
                self.offset = exc_value.offset
                self.end_offset = exc_value.end_offset
                self.msg = exc_value.msg
            elif exc_type and issubclass(exc_type, ImportError) and \
                    getattr(exc_value, "name_from", None) is not None:
                wrong_name = getattr(exc_value, "name_from", None)
                suggestion = _compute_suggestion_error(exc_value, exc_traceback, wrong_name)
                if suggestion:
                    self._str += f". Did you mean: '{suggestion}'?"
            elif exc_type and issubclass(exc_type, (NameError, AttributeError)) and \
                    getattr(exc_value, "name", None) is not None:
                wrong_name = getattr(exc_value, "name", None)
                suggestion = _compute_suggestion_error(exc_value, exc_traceback, wrong_name)
                if suggestion:
                    self._str += f". Did you mean: '{suggestion}'?"
                if issubclass(exc_type, NameError):
                    wrong_name = getattr(exc_value, "name", None)
                    if wrong_name is not None and wrong_name in sys.stdlib_module_names:
                        if suggestion:
                            self._str += f" Or did you forget to import '{wrong_name}'"
                        else:
                            self._str += f". Did you forget to import '{wrong_name}'"
            if lookup_lines:
                self._load_lines()
            self.__suppress_context__ = \
                exc_value.__suppress_context__ if exc_value is not None else False

            # Convert __cause__ and __context__ to `TracebackExceptions`s, use a
            # queue to avoid recursion (only the top-level call gets _seen == None)
            if not is_recursive_call:
                queue = [(self, exc_value)]
                while queue:
                    te, e = queue.pop()
                    if (e and e.__cause__ is not None
                        and id(e.__cause__) not in _seen):
                        cause = TracebackException(
                            type(e.__cause__),
                            e.__cause__,
                            e.__cause__.__traceback__,
                            limit=limit,
                            lookup_lines=lookup_lines,
                            capture_locals=capture_locals,
                            max_group_width=max_group_width,
                            max_group_depth=max_group_depth,
                            _seen=_seen)
                    else:
                        cause = None

                    if compact:
                        need_context = (cause is None and
                                        e is not None and
                                        not e.__suppress_context__)
                    else:
                        need_context = True
                    if (e and e.__context__ is not None
                        and need_context and id(e.__context__) not in _seen):
                        context = TracebackException(
                            type(e.__context__),
                            e.__context__,
                            e.__context__.__traceback__,
                            limit=limit,
                            lookup_lines=lookup_lines,
                            capture_locals=capture_locals,
                            max_group_width=max_group_width,
                            max_group_depth=max_group_depth,
                            _seen=_seen)
                    else:
                        context = None

                    if e and isinstance(e, BaseExceptionGroup):
                        exceptions = []
                        for exc in e.exceptions:
                            texc = TracebackException(
                                type(exc),
                                exc,
                                exc.__traceback__,
                                limit=limit,
                                lookup_lines=lookup_lines,
                                capture_locals=capture_locals,
                                max_group_width=max_group_width,
                                max_group_depth=max_group_depth,
                                _seen=_seen)
                            exceptions.append(texc)
                    else:
                        exceptions = None

                    te.__cause__ = cause
                    te.__context__ = context
                    te.exceptions = exceptions
                    if cause:
                        queue.append((te.__cause__, e.__cause__))
                    if context:
                        queue.append((te.__context__, e.__context__))
                    if exceptions:
                        queue.extend(zip(te.exceptions, e.exceptions))

        @classmethod
        def from_exception(cls, exc, *args, **kwargs):
            """Create a TracebackException from an exception."""
            return cls(type(exc), exc, exc.__traceback__, *args, **kwargs)

        def _load_lines(self):
            """Private API. force all lines in the stack to be loaded."""
            for frame in self.stack:
                frame.line

        def __eq__(self, other):
            if isinstance(other, TracebackException):
                return self.__dict__ == other.__dict__
            return NotImplemented

        def __str__(self):
            return self._str

        def format_exception_only(self):
            """Format the exception part of the traceback.

            The return value is a generator of strings, each ending in a newline.

            Generator yields the exception message.
            For :exc:`SyntaxError` exceptions, it
            also yields (before the exception message)
            several lines that (when printed)
            display detailed information about where the syntax error occurred.
            Following the message, generator also yields
            all the exception's ``__notes__``.
            """
            if self.exc_type is None:
                yield _format_final_exc_line(None, self._str)
                return

            stype = self.exc_type.__qualname__
            smod = self.exc_type.__module__
            if smod not in ("__main__", "builtins"):
                if not isinstance(smod, str):
                    smod = "<unknown>"
                stype = smod + '.' + stype

            if not issubclass(self.exc_type, SyntaxError):
                yield _format_final_exc_line(stype, self._str)
            else:
                yield from self._format_syntax_error(stype)

            if (
                isinstance(self.__notes__, _abc.Sequence)
                and not isinstance(self.__notes__, (str, bytes))
            ):
                for note in self.__notes__:
                    note = _safe_string(note, 'note')
                    yield from [l + '\n' for l in note.split('\n')]
            elif self.__notes__ is not None:
                yield "{}\n".format(_safe_string(self.__notes__, '__notes__', func=repr))

        def _format_syntax_error(self, stype):
            """Format SyntaxError exceptions (internal helper)."""
            # Show exactly where the problem was found.
            filename_suffix = ''
            if self.lineno is not None:
                yield '  File "{}", line {}\n'.format(
                    self.filename or "<string>", self.lineno)
            elif self.filename is not None:
                filename_suffix = ' ({})'.format(self.filename)

            text = self.text
            if text is not None:
                # text  = "   foo\n"
                # rtext = "   foo"
                # ltext =    "foo"
                rtext = text.rstrip('\n')
                ltext = rtext.lstrip(' \n\f')
                spaces = len(rtext) - len(ltext)
                yield '    {}\n'.format(ltext)

                if self.offset is not None:
                    offset = self.offset
                    end_offset = self.end_offset if self.end_offset not in {None, 0} else offset
                    if offset == end_offset or end_offset == -1:
                        end_offset = offset + 1

                    # Convert 1-based column offset to 0-based index into stripped text
                    colno = offset - 1 - spaces
                    end_colno = end_offset - 1 - spaces
                    if colno >= 0:
                        # non-space whitespace (likes tabs) must be kept for alignment
                        caretspace = ((c if c.isspace() else ' ') for c in ltext[:colno])
                        yield '    {}{}'.format("".join(caretspace), ('^' * (end_colno - colno) + "\n"))
            msg = self.msg or "<no detail available>"
            yield "{}: {}{}\n".format(stype, msg, filename_suffix)

        def format(self, *, chain=True, _ctx=None):
            """Format the exception.

            If chain is not *True*, *__cause__* and *__context__* will not be formatted.

            The return value is a generator of strings, each ending in a newline and
            some containing internal newlines. `print_exception` is a wrapper around
            this method which just prints the lines to a file.

            The message indicating which exception occurred is always the last
            string in the output.
            """

            if _ctx is None:
                _ctx = _ExceptionPrintContext()

            output = []
            exc = self
            if chain:
                while exc:
                    if exc.__cause__ is not None:
                        chained_msg = _cause_message
                        chained_exc = exc.__cause__
                    elif (exc.__context__  is not None and
                        not exc.__suppress_context__):
                        chained_msg = _context_message
                        chained_exc = exc.__context__
                    else:
                        chained_msg = None
                        chained_exc = None

                    output.append((chained_msg, exc))
                    exc = chained_exc
            else:
                output.append((None, exc))

            for msg, exc in reversed(output):
                if msg is not None:
                    yield from _ctx.emit(msg)
                if exc.exceptions is None:
                    if exc.stack:
                        yield from _ctx.emit('Traceback (most recent call last):\n')
                        yield from _ctx.emit(exc.stack.format())
                    yield from _ctx.emit(exc.format_exception_only())
                elif _ctx.exception_group_depth > self.max_group_depth:
                    # exception group, but depth exceeds limit
                    yield from _ctx.emit(
                        f"... (max_group_depth is {self.max_group_depth})\n")
                else:
                    # format exception group
                    is_toplevel = (_ctx.exception_group_depth == 0)
                    if is_toplevel:
                        _ctx.exception_group_depth += 1

                    if exc.stack:
                        yield from _ctx.emit(
                            'Exception Group Traceback (most recent call last):\n',
                            margin_char = '+' if is_toplevel else None)
                        yield from _ctx.emit(exc.stack.format())

                    yield from _ctx.emit(exc.format_exception_only())
                    num_excs = len(exc.exceptions)
                    if num_excs <= self.max_group_width:
                        n = num_excs
                    else:
                        n = self.max_group_width + 1
                    _ctx.need_close = False
                    for i in range(n):
                        last_exc = (i == n-1)
                        if last_exc:
                            # The closing frame may be added by a recursive call
                            _ctx.need_close = True

                        if self.max_group_width is not None:
                            truncated = (i >= self.max_group_width)
                        else:
                            truncated = False
                        title = f'{i+1}' if not truncated else '...'
                        yield (_ctx.indent() +
                            ('+-' if i==0 else '  ') +
                            f'+---------------- {title} ----------------\n')
                        _ctx.exception_group_depth += 1
                        if not truncated:
                            yield from exc.exceptions[i].format(chain=chain, _ctx=_ctx)
                        else:
                            remaining = num_excs - self.max_group_width
                            plural = 's' if remaining > 1 else ''
                            yield from _ctx.emit(
                                f"and {remaining} more exception{plural}\n")

                        if last_exc and _ctx.need_close:
                            yield (_ctx.indent() +
                                "+------------------------------------\n")
                            _ctx.need_close = False
                        _ctx.exception_group_depth -= 1

                    if is_toplevel:
                        assert _ctx.exception_group_depth == 1
                        _ctx.exception_group_depth = 0


        def print(self, *, file=None, chain=True):
            """Print the result of self.format(chain=chain) to 'file'."""
            if file is None:
                file = sys.stderr
            for line in self.format(chain=chain):
                print(line, file=file, end="")

    def _print_exception(exc, /, value=_sentinel, tb=_sentinel, limit=None, \
                    file=None, chain=True):
        """Print exception up to 'limit' stack trace entries from 'tb' to 'file'.

        This differs from print_tb() in the following ways: (1) if
        traceback is not None, it prints a header "Traceback (most recent
        call last):"; (2) it prints the exception type and value after the
        stack trace; (3) if type is SyntaxError and value has the
        appropriate format, it prints the line where the syntax error
        occurred with a caret on the next line indicating the approximate
        position of the error.
        """
        value, tb = _parse_value_tb(exc, value, tb)
        te = TracebackException(type(value), value, tb, limit=limit, compact=True)
        te.print(file=file, chain=chain)
    
    from collections import namedtuple

    _ExceptHookArgs = namedtuple(
        'ExceptHookArgs',
        'exc_type exc_value exc_traceback thread')

    def ExceptHookArgs(args):
        return _ExceptHookArgs(*args)

    def excepthook(args, /):
        """
        Handle uncaught Thread.run() exception.
        """
        if args.exc_type == SystemExit:
            # silently ignore SystemExit
            return

        if _sys is not None and _sys.stderr is not None:
            stderr = _sys.stderr
        elif args.thread is not None:
            stderr = args.thread._stderr
            if stderr is None:
                # do nothing if sys.stderr is None and sys.stderr was None
                # when the thread was created
                return
        else:
            # do nothing if sys.stderr is None and args.thread is None
            return

        if args.thread is not None:
            name = args.thread.name
        else:
            name = get_ident()
        print(f"Exception in thread {name}:",
              file=stderr, flush=True)
        _print_exception(args.exc_type, args.exc_value, args.exc_traceback,
                         file=stderr)
        stderr.flush()

# types.py
GenericAlias = type(list[int])

# _weakrefset.py

class _IterationGuard:
    # This context manager registers itself in the current iterators of the
    # weak container, such as to delay all removals until the context manager
    # exits.
    # This technique should be relatively thread-safe (since sets are).

    def __init__(self, weakcontainer):
        # Don't create cycles
        self.weakcontainer = ref(weakcontainer)

    def __enter__(self):
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()

class WeakSet:
    def __init__(self, data=None):
        self.data = set()
        def _remove(item, selfref=ref(self)):
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(item)
                else:
                    self.data.discard(item)
        self._remove = _remove
        # A list of keys to be removed
        self._pending_removals = []
        self._iterating = set()
        if data is not None:
            self.update(data)

    def _commit_removals(self):
        pop = self._pending_removals.pop
        discard = self.data.discard
        while True:
            try:
                item = pop()
            except IndexError:
                return
            discard(item)

    def __iter__(self):
        with _IterationGuard(self):
            for itemref in self.data:
                item = itemref()
                if item is not None:
                    # Caveat: the iterator will keep a strong reference to
                    # `item` until it is resumed or closed.
                    yield item

    def __len__(self):
        return len(self.data) - len(self._pending_removals)

    def __contains__(self, item):
        try:
            wr = ref(item)
        except TypeError:
            return False
        return wr in self.data

    def __reduce__(self):
        return self.__class__, (list(self),), self.__getstate__()

    def add(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.add(ref(item, self._remove))

    def clear(self):
        if self._pending_removals:
            self._commit_removals()
        self.data.clear()

    def copy(self):
        return self.__class__(self)

    def pop(self):
        if self._pending_removals:
            self._commit_removals()
        while True:
            try:
                itemref = self.data.pop()
            except KeyError:
                raise KeyError('pop from empty WeakSet') from None
            item = itemref()
            if item is not None:
                return item

    def remove(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.remove(ref(item))

    def discard(self, item):
        if self._pending_removals:
            self._commit_removals()
        self.data.discard(ref(item))

    def update(self, other):
        if self._pending_removals:
            self._commit_removals()
        for element in other:
            self.add(element)

    def __ior__(self, other):
        self.update(other)
        return self

    def difference(self, other):
        newset = self.copy()
        newset.difference_update(other)
        return newset
    __sub__ = difference

    def difference_update(self, other):
        self.__isub__(other)
    def __isub__(self, other):
        if self._pending_removals:
            self._commit_removals()
        if self is other:
            self.data.clear()
        else:
            self.data.difference_update(ref(item) for item in other)
        return self

    def intersection(self, other):
        return self.__class__(item for item in other if item in self)
    __and__ = intersection

    def intersection_update(self, other):
        self.__iand__(other)
    def __iand__(self, other):
        if self._pending_removals:
            self._commit_removals()
        self.data.intersection_update(ref(item) for item in other)
        return self

    def issubset(self, other):
        return self.data.issubset(ref(item) for item in other)
    __le__ = issubset

    def __lt__(self, other):
        return self.data < set(map(ref, other))

    def issuperset(self, other):
        return self.data.issuperset(ref(item) for item in other)
    __ge__ = issuperset

    def __gt__(self, other):
        return self.data > set(map(ref, other))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.data == set(map(ref, other))

    def symmetric_difference(self, other):
        newset = self.copy()
        newset.symmetric_difference_update(other)
        return newset
    __xor__ = symmetric_difference

    def symmetric_difference_update(self, other):
        self.__ixor__(other)
    def __ixor__(self, other):
        if self._pending_removals:
            self._commit_removals()
        if self is other:
            self.data.clear()
        else:
            self.data.symmetric_difference_update(ref(item, self._remove) for item in other)
        return self

    def union(self, other):
        return self.__class__(e for s in (self, other) for e in s)
    __or__ = union

    def isdisjoint(self, other):
        return len(self.intersection(other)) == 0

    def __repr__(self):
        return repr(self.data)

    __class_getitem__ = classmethod(GenericAlias)


# random.py threading.py
# Original value of threading.excepthook
__excepthook__ = excepthook

_dangling = WeakSet()


def _make_invoke_excepthook():
    # Create a local namespace to ensure that variables remain alive
    # when _invoke_excepthook() is called, even if it is called late during
    # Python shutdown. It is mostly needed for daemon threads.

    old_excepthook = excepthook
    old_sys_excepthook = _sys.excepthook
    if old_excepthook is None:
        raise RuntimeError("threading.excepthook is None")
    if old_sys_excepthook is None:
        raise RuntimeError("sys.excepthook is None")

    sys_exc_info = _sys.exc_info
    local_print = print
    local_sys = _sys

    def invoke_excepthook(thread):
        global excepthook
        try:
            hook = excepthook
            if hook is None:
                hook = old_excepthook

            args = ExceptHookArgs([*sys_exc_info(), thread])

            hook(args)
        except Exception as exc:
            exc.__suppress_context__ = True
            del exc

            if local_sys is not None and local_sys.stderr is not None:
                stderr = local_sys.stderr
            else:
                stderr = thread._stderr

            local_print("Exception in threading.excepthook:",
                        file=stderr, flush=True)

            if local_sys is not None and local_sys.excepthook is not None:
                sys_excepthook = local_sys.excepthook
            else:
                sys_excepthook = old_sys_excepthook

            sys_excepthook(*sys_exc_info())
        finally:
            # Break reference cycle (exception stored in a variable)
            args = None

    return invoke_excepthook


Lock = _allocate_lock

def RLock(*args, **kwargs):
    """Factory function that returns a new reentrant lock.

    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it again
    without blocking; the thread must release it once for each time it has
    acquired it.

    """
    if _CRLock is None:
        return _PyRLock(*args, **kwargs)
    return _CRLock(*args, **kwargs)

class _RLock:
    """This class implements reentrant lock objects.

    A reentrant lock must be released by the thread that acquired it. Once a
    thread has acquired a reentrant lock, the same thread may acquire it
    again without blocking; the thread must release it once for each time it
    has acquired it.

    """

    def __init__(self):
        self._block = _allocate_lock()
        self._owner = None
        self._count = 0

    def __repr__(self):
        owner = self._owner
        try:
            owner = _active[owner].name
        except KeyError:
            pass
        return "<%s %s.%s object owner=%r count=%d at %s>" % (
            "locked" if self._block.locked() else "unlocked",
            self.__class__.__module__,
            self.__class__.__qualname__,
            owner,
            self._count,
            hex(id(self))
        )

    def _at_fork_reinit(self):
        self._block._at_fork_reinit()
        self._owner = None
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        """Acquire a lock, blocking or non-blocking.

        When invoked without arguments: if this thread already owns the lock,
        increment the recursion level by one, and return immediately. Otherwise,
        if another thread owns the lock, block until the lock is unlocked. Once
        the lock is unlocked (not owned by any thread), then grab ownership, set
        the recursion level to one, and return. If more than one thread is
        blocked waiting until the lock is unlocked, only one at a time will be
        able to grab ownership of the lock. There is no return value in this
        case.

        When invoked with the blocking argument set to true, do the same thing
        as when called without arguments, and return true.

        When invoked with the blocking argument set to false, do not block. If a
        call without an argument would block, return false immediately;
        otherwise, do the same thing as when called without arguments, and
        return true.

        When invoked with the floating-point timeout argument set to a positive
        value, block for at most the number of seconds specified by timeout
        and as long as the lock cannot be acquired.  Return true if the lock has
        been acquired, false if the timeout has elapsed.

        """
        me = get_ident()
        if self._owner == me:
            self._count += 1
            return 1
        rc = self._block.acquire(blocking, timeout)
        if rc:
            self._owner = me
            self._count = 1
        return rc

    __enter__ = acquire

    def release(self):
        """Release a lock, decrementing the recursion level.

        If after the decrement it is zero, reset the lock to unlocked (not owned
        by any thread), and if any other threads are blocked waiting for the
        lock to become unlocked, allow exactly one of them to proceed. If after
        the decrement the recursion level is still nonzero, the lock remains
        locked and owned by the calling thread.

        Only call this method when the calling thread owns the lock. A
        RuntimeError is raised if this method is called when the lock is
        unlocked.

        There is no return value.

        """
        if self._owner != get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._count = count = self._count - 1
        if not count:
            self._owner = None
            self._block.release()

    def __exit__(self, t, v, tb):
        self.release()

    # Internal methods used by condition variables

    def _acquire_restore(self, state):
        self._block.acquire()
        self._count, self._owner = state

    def _release_save(self):
        if self._count == 0:
            raise RuntimeError("cannot release un-acquired lock")
        count = self._count
        self._count = 0
        owner = self._owner
        self._owner = None
        self._block.release()
        return (count, owner)

    def _is_owned(self):
        return self._owner == get_ident()

    # Internal method used for reentrancy checks

    def _recursion_count(self):
        if self._owner != get_ident():
            return 0
        return self._count

_PyRLock = _RLock
_active_limbo_lock = RLock()

class Condition:
    """Class that implements a condition variable.

    A condition variable allows one or more threads to wait until they are
    notified by another thread.

    If the lock argument is given and not None, it must be a Lock or RLock
    object, and it is used as the underlying lock. Otherwise, a new RLock object
    is created and used as the underlying lock.

    """

    def __init__(self, lock=None):
        if lock is None:
            lock = RLock()
        self._lock = lock
        # Export the lock's acquire() and release() methods
        self.acquire = lock.acquire
        self.release = lock.release
        # If the lock defines _release_save() and/or _acquire_restore(),
        # these override the default implementations (which just call
        # release() and acquire() on the lock).  Ditto for _is_owned().
        if hasattr(lock, '_release_save'):
            self._release_save = lock._release_save
        if hasattr(lock, '_acquire_restore'):
            self._acquire_restore = lock._acquire_restore
        if hasattr(lock, '_is_owned'):
            self._is_owned = lock._is_owned
        self._waiters = _deque()

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()
        self._waiters.clear()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __repr__(self):
        return "<Condition(%s, %d)>" % (self._lock, len(self._waiters))

    def _release_save(self):
        self._lock.release()           # No state to save

    def _acquire_restore(self, x):
        self._lock.acquire()           # Ignore saved state

    def _is_owned(self):
        # Return True if lock is owned by current_thread.
        # This method is called only if _lock doesn't have _is_owned().
        if self._lock.acquire(False):
            self._lock.release()
            return False
        else:
            return True

    def wait(self, timeout=None):
        """Wait until notified or until a timeout occurs.

        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.

        This method releases the underlying lock, and then blocks until it is
        awakened by a notify() or notify_all() call for the same condition
        variable in another thread, or until the optional timeout occurs. Once
        awakened or timed out, it re-acquires the lock and returns.

        When the timeout argument is present and not None, it should be a
        floating-point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        When the underlying lock is an RLock, it is not released using its
        release() method, since this may not actually unlock the lock when it
        was acquired multiple times recursively. Instead, an internal interface
        of the RLock class is used, which really unlocks it even when it has
        been recursively acquired several times. Another internal interface is
        then used to restore the recursion level when the lock is reacquired.

        """
        if not self._is_owned():
            raise RuntimeError("cannot wait on un-acquired lock")
        waiter = _allocate_lock()
        waiter.acquire()
        self._waiters.append(waiter)
        saved_state = self._release_save()
        gotit = False
        try:    # restore state no matter what (e.g., KeyboardInterrupt)
            if timeout is None:
                waiter.acquire()
                gotit = True
            else:
                if timeout > 0:
                    gotit = waiter.acquire(True, timeout)
                else:
                    gotit = waiter.acquire(False)
            return gotit
        finally:
            self._acquire_restore(saved_state)
            if not gotit:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass

    def wait_for(self, predicate, timeout=None):
        """Wait until a condition evaluates to True.

        predicate should be a callable which result will be interpreted as a
        boolean value.  A timeout may be provided giving the maximum time to
        wait.

        """
        endtime = None
        waittime = timeout
        result = predicate()
        while not result:
            if waittime is not None:
                if endtime is None:
                    endtime = _time() + waittime
                else:
                    waittime = endtime - _time()
                    if waittime <= 0:
                        break
            self.wait(waittime)
            result = predicate()
        return result

    def notify(self, n=1):
        """Wake up one or more threads waiting on this condition, if any.

        If the calling thread has not acquired the lock when this method is
        called, a RuntimeError is raised.

        This method wakes up at most n of the threads waiting for the condition
        variable; it is a no-op if no threads are waiting.

        """
        if not self._is_owned():
            raise RuntimeError("cannot notify on un-acquired lock")
        waiters = self._waiters
        while waiters and n > 0:
            waiter = waiters[0]
            try:
                waiter.release()
            except RuntimeError:
                # gh-92530: The previous call of notify() released the lock,
                # but was interrupted before removing it from the queue.
                # It can happen if a signal handler raises an exception,
                # like CTRL+C which raises KeyboardInterrupt.
                pass
            else:
                n -= 1
            try:
                waiters.remove(waiter)
            except ValueError:
                pass

    def notify_all(self):
        """Wake up all threads waiting on this condition.

        If the calling thread has not acquired the lock when this method
        is called, a RuntimeError is raised.

        """
        self.notify(len(self._waiters))

    def notifyAll(self):
        """Wake up all threads waiting on this condition.

        This method is deprecated, use notify_all() instead.

        """
        import warnings
        warnings.warn('notifyAll() is deprecated, use notify_all() instead',
                      DeprecationWarning, stacklevel=2)
        self.notify_all()

class Event:
    """Class implementing event objects.

    Events manage a flag that can be set to true with the set() method and reset
    to false with the clear() method. The wait() method blocks until the flag is
    true.  The flag is initially false.

    """

    # After Tim Peters' event class (without is_posted())

    def __init__(self):
        self._cond = Condition(Lock())
        self._flag = False

    def __repr__(self):
        cls = self.__class__
        status = 'set' if self._flag else 'unset'
        return f"<{cls.__module__}.{cls.__qualname__} at {id(self):#x}: {status}>"

    def _at_fork_reinit(self):
        # Private method called by Thread._reset_internal_locks()
        self._cond._at_fork_reinit()

    def is_set(self):
        """Return true if and only if the internal flag is true."""
        return self._flag

    def isSet(self):
        """Return true if and only if the internal flag is true.

        This method is deprecated, use is_set() instead.

        """
        import warnings
        warnings.warn('isSet() is deprecated, use is_set() instead',
                      DeprecationWarning, stacklevel=2)
        return self.is_set()

    def set(self):
        """Set the internal flag to true.

        All threads waiting for it to become true are awakened. Threads
        that call wait() once the flag is true will not block at all.

        """
        with self._cond:
            self._flag = True
            self._cond.notify_all()

    def clear(self):
        """Reset the internal flag to false.

        Subsequently, threads calling wait() will block until set() is called to
        set the internal flag to true again.

        """
        with self._cond:
            self._flag = False

    def wait(self, timeout=None):
        """Block until the internal flag is true.

        If the internal flag is true on entry, return immediately. Otherwise,
        block until another thread calls set() to set the flag to true, or until
        the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating-point number specifying a timeout for the operation in seconds
        (or fractions thereof).

        This method returns the internal flag on exit, so it will always return
        True except if a timeout is given and the operation times out.

        """
        with self._cond:
            signaled = self._flag
            if not signaled:
                signaled = self._cond.wait(timeout)
            return signaled

def _maintain_shutdown_locks():
    """
    Drop any shutdown locks that don't correspond to running threads anymore.

    Calling this from time to time avoids an ever-growing _shutdown_locks
    set when Thread objects are not joined explicitly. See bpo-37788.

    This must be called with _shutdown_locks_lock acquired.
    """
    # If a lock was released, the corresponding thread has exited
    to_remove = [lock for lock in _shutdown_locks if not lock.locked()]
    _shutdown_locks.difference_update(to_remove)

# Helper to generate new thread names
_counter = _count(1).__next__
def _newname(name_template):
    return name_template % _counter()

def current_thread():
    """Return the current Thread object, corresponding to the caller's thread of control.

    If the caller's thread of control was not created through the threading
    module, a dummy thread object with limited functionality is returned.

    """
    try:
        return _active[get_ident()]
    except KeyError:
        return _DummyThread()

def getrandbits(k):
        """getrandbits(k) -> x.  Generates an int with k random bits."""
        if k < 0:
            raise ValueError('number of bits must be non-negative')
        numbytes = (k + 7) // 8                       # bits / 8 and rounded up
        x = int.from_bytes(_urandom(numbytes))
        return x >> (numbytes * 8 - k)                # trim excess bits

def _randbelow_with_getrandbits(n):
        "Return a random int in the range [0,n).  Defined for n > 0."

        k = n.bit_length()
        r = getrandbits(k)  # 0 <= r < 2**k
        while r >= n:
            r = getrandbits(k)
        return r

_randbelow = _randbelow_with_getrandbits

def shuffle(x):
        """Shuffle list x in place, and return None."""

        randbelow = _randbelow
        for i in reversed(range(1, len(x))):
            # pick an element in x[:i+1] with which to exchange x[i]
            j = randbelow(i + 1)
            x[i], x[j] = x[j], x[i]

class Thread:
    """A class that represents a thread of control.

    This class can be safely subclassed in a limited fashion. There are two ways
    to specify the activity: by passing a callable object to the constructor, or
    by overriding the run() method in a subclass.

    """

    _initialized = False

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        """This constructor should always be called with keyword arguments. Arguments are:

        *group* should be None; reserved for future extension when a ThreadGroup
        class is implemented.

        *target* is the callable object to be invoked by the run()
        method. Defaults to None, meaning nothing is called.

        *name* is the thread name. By default, a unique name is constructed of
        the form "Thread-N" where N is a small decimal number.

        *args* is a list or tuple of arguments for the target invocation. Defaults to ().

        *kwargs* is a dictionary of keyword arguments for the target
        invocation. Defaults to {}.

        If a subclass overrides the constructor, it must make sure to invoke
        the base class constructor (Thread.__init__()) before doing anything
        else to the thread.

        """
        assert group is None, "group argument must be None for now"
        if kwargs is None:
            kwargs = {}
        if name:
            name = str(name)
        else:
            name = _newname("Thread-%d")
            if target is not None:
                try:
                    target_name = target.__name__
                    name += f" ({target_name})"
                except AttributeError:
                    pass

        self._target = target
        self._name = name
        self._args = args
        self._kwargs = kwargs
        if daemon is not None:
            if daemon and not _daemon_threads_allowed():
                raise RuntimeError('daemon threads are disabled in this (sub)interpreter')
            self._daemonic = daemon
        else:
            self._daemonic = current_thread().daemon
        self._ident = None
        if _HAVE_THREAD_NATIVE_ID:
            self._native_id = None
        self._tstate_lock = None
        self._started = Event()
        self._is_stopped = False
        self._initialized = True
        # Copy of sys.stderr used by self._invoke_excepthook()
        self._stderr = _sys.stderr
        self._invoke_excepthook = _make_invoke_excepthook()
        # For debugging and _after_fork()
        _dangling.add(self)

    def _reset_internal_locks(self, is_alive):
        # private!  Called by _after_fork() to reset our internal locks as
        # they may be in an invalid state leading to a deadlock or crash.
        self._started._at_fork_reinit()
        if is_alive:
            # bpo-42350: If the fork happens when the thread is already stopped
            # (ex: after threading._shutdown() has been called), _tstate_lock
            # is None. Do nothing in this case.
            if self._tstate_lock is not None:
                self._tstate_lock._at_fork_reinit()
                self._tstate_lock.acquire()
        else:
            # The thread isn't alive after fork: it doesn't have a tstate
            # anymore.
            self._is_stopped = True
            self._tstate_lock = None

    def __repr__(self):
        assert self._initialized, "Thread.__init__() was not called"
        status = "initial"
        if self._started.is_set():
            status = "started"
        self.is_alive() # easy way to get ._is_stopped set when appropriate
        if self._is_stopped:
            status = "stopped"
        if self._daemonic:
            status += " daemon"
        if self._ident is not None:
            status += " %s" % self._ident
        return "<%s(%s, %s)>" % (self.__class__.__name__, self._name, status)

    def start(self):
        """Start the thread's activity.

        It must be called at most once per thread object. It arranges for the
        object's run() method to be invoked in a separate thread of control.

        This method will raise a RuntimeError if called more than once on the
        same thread object.

        """
        if not self._initialized:
            raise RuntimeError("thread.__init__() not called")

        if self._started.is_set():
            raise RuntimeError("threads can only be started once")

        with _active_limbo_lock:
            _limbo[self] = self
        try:
            _start_new_thread(self._bootstrap, ())
        except Exception:
            with _active_limbo_lock:
                del _limbo[self]
            raise
        self._started.wait()

    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.

        """
        try:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def _bootstrap(self):
        # Wrapper around the real bootstrap code that ignores
        # exceptions during interpreter cleanup.  Those typically
        # happen when a daemon thread wakes up at an unfortunate
        # moment, finds the world around it destroyed, and raises some
        # random exception *** while trying to report the exception in
        # _bootstrap_inner() below ***.  Those random exceptions
        # don't help anybody, and they confuse users, so we suppress
        # them.  We suppress them only when it appears that the world
        # indeed has already been destroyed, so that exceptions in
        # _bootstrap_inner() during normal business hours are properly
        # reported.  Also, we only suppress them for daemonic threads;
        # if a non-daemonic encounters this, something else is wrong.
        try:
            self._bootstrap_inner()
        except:
            if self._daemonic and _sys is None:
                return
            raise

    def _set_ident(self):
        self._ident = get_ident()

    if _HAVE_THREAD_NATIVE_ID:
        def _set_native_id(self):
            self._native_id = get_native_id()

    def _set_tstate_lock(self):
        """
        Set a lock object which will be released by the interpreter when
        the underlying thread state (see pystate.h) gets deleted.
        """
        self._tstate_lock = _set_sentinel()
        self._tstate_lock.acquire()

        if not self.daemon:
            with _shutdown_locks_lock:
                _maintain_shutdown_locks()
                _shutdown_locks.add(self._tstate_lock)

    def _bootstrap_inner(self):
        try:
            self._set_ident()
            self._set_tstate_lock()
            if _HAVE_THREAD_NATIVE_ID:
                self._set_native_id()
            self._started.set()
            with _active_limbo_lock:
                _active[self._ident] = self
                del _limbo[self]

            if _trace_hook:
                _sys.settrace(_trace_hook)
            if _profile_hook:
                _sys.setprofile(_profile_hook)

            try:
                self.run()
            except:
                self._invoke_excepthook(self)
        finally:
            self._delete()

    def _stop(self):
        # After calling ._stop(), .is_alive() returns False and .join() returns
        # immediately.  ._tstate_lock must be released before calling ._stop().
        #
        # Normal case:  C code at the end of the thread's life
        # (release_sentinel in _threadmodule.c) releases ._tstate_lock, and
        # that's detected by our ._wait_for_tstate_lock(), called by .join()
        # and .is_alive().  Any number of threads _may_ call ._stop()
        # simultaneously (for example, if multiple threads are blocked in
        # .join() calls), and they're not serialized.  That's harmless -
        # they'll just make redundant rebindings of ._is_stopped and
        # ._tstate_lock.  Obscure:  we rebind ._tstate_lock last so that the
        # "assert self._is_stopped" in ._wait_for_tstate_lock() always works
        # (the assert is executed only if ._tstate_lock is None).
        #
        # Special case:  _main_thread releases ._tstate_lock via this
        # module's _shutdown() function.
        lock = self._tstate_lock
        if lock is not None:
            assert not lock.locked()
        self._is_stopped = True
        self._tstate_lock = None
        if not self.daemon:
            with _shutdown_locks_lock:
                # Remove our lock and other released locks from _shutdown_locks
                _maintain_shutdown_locks()

    def _delete(self):
        "Remove current thread from the dict of currently running threads."
        with _active_limbo_lock:
            del _active[get_ident()]
            # There must not be any python code between the previous line
            # and after the lock is released.  Otherwise a tracing function
            # could try to acquire the lock again in the same thread, (in
            # current_thread()), and would block.

    def join(self, timeout=None):
        """Wait until the thread terminates.

        This blocks the calling thread until the thread whose join() method is
        called terminates -- either normally or through an unhandled exception
        or until the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating-point number specifying a timeout for the operation in seconds
        (or fractions thereof). As join() always returns None, you must call
        is_alive() after join() to decide whether a timeout happened -- if the
        thread is still alive, the join() call timed out.

        When the timeout argument is not present or None, the operation will
        block until the thread terminates.

        A thread can be join()ed many times.

        join() raises a RuntimeError if an attempt is made to join the current
        thread as that would cause a deadlock. It is also an error to join() a
        thread before it has been started and attempts to do so raises the same
        exception.

        """
        if not self._initialized:
            raise RuntimeError("Thread.__init__() not called")
        if not self._started.is_set():
            raise RuntimeError("cannot join thread before it is started")
        if self is current_thread():
            raise RuntimeError("cannot join current thread")

        if timeout is None:
            self._wait_for_tstate_lock()
        else:
            # the behavior of a negative timeout isn't documented, but
            # historically .join(timeout=x) for x<0 has acted as if timeout=0
            self._wait_for_tstate_lock(timeout=max(timeout, 0))

    def _wait_for_tstate_lock(self, block=True, timeout=-1):
        # Issue #18808: wait for the thread state to be gone.
        # At the end of the thread's life, after all knowledge of the thread
        # is removed from C data structures, C code releases our _tstate_lock.
        # This method passes its arguments to _tstate_lock.acquire().
        # If the lock is acquired, the C code is done, and self._stop() is
        # called.  That sets ._is_stopped to True, and ._tstate_lock to None.
        lock = self._tstate_lock
        if lock is None:
            # already determined that the C code is done
            assert self._is_stopped
            return

        try:
            if lock.acquire(block, timeout):
                lock.release()
                self._stop()
        except:
            if lock.locked():
                # bpo-45274: lock.acquire() acquired the lock, but the function
                # was interrupted with an exception before reaching the
                # lock.release(). It can happen if a signal handler raises an
                # exception, like CTRL+C which raises KeyboardInterrupt.
                lock.release()
                self._stop()
            raise

    @property
    def name(self):
        """A string used for identification purposes only.

        It has no semantics. Multiple threads may be given the same name. The
        initial name is set by the constructor.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._name

    @name.setter
    def name(self, name):
        assert self._initialized, "Thread.__init__() not called"
        self._name = str(name)

    @property
    def ident(self):
        """Thread identifier of this thread or None if it has not been started.

        This is a nonzero integer. See the get_ident() function. Thread
        identifiers may be recycled when a thread exits and another thread is
        created. The identifier is available even after the thread has exited.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._ident

    if _HAVE_THREAD_NATIVE_ID:
        @property
        def native_id(self):
            """Native integral thread ID of this thread, or None if it has not been started.

            This is a non-negative integer. See the get_native_id() function.
            This represents the Thread ID as reported by the kernel.

            """
            assert self._initialized, "Thread.__init__() not called"
            return self._native_id

    def is_alive(self):
        """Return whether the thread is alive.

        This method returns True just before the run() method starts until just
        after the run() method terminates. See also the module function
        enumerate().

        """
        assert self._initialized, "Thread.__init__() not called"
        if self._is_stopped or not self._started.is_set():
            return False
        self._wait_for_tstate_lock(False)
        return not self._is_stopped

    @property
    def daemon(self):
        """A boolean value indicating whether this thread is a daemon thread.

        This must be set before start() is called, otherwise RuntimeError is
        raised. Its initial value is inherited from the creating thread; the
        main thread is not a daemon thread and therefore all threads created in
        the main thread default to daemon = False.

        The entire Python program exits when only daemon threads are left.

        """
        assert self._initialized, "Thread.__init__() not called"
        return self._daemonic

    @daemon.setter
    def daemon(self, daemonic):
        if not self._initialized:
            raise RuntimeError("Thread.__init__() not called")
        if daemonic and not _daemon_threads_allowed():
            raise RuntimeError('daemon threads are disabled in this interpreter')
        if self._started.is_set():
            raise RuntimeError("cannot set daemon status of active thread")
        self._daemonic = daemonic

    def isDaemon(self):
        """Return whether this thread is a daemon.

        This method is deprecated, use the daemon attribute instead.

        """
        import warnings
        warnings.warn('isDaemon() is deprecated, get the daemon attribute instead',
                      DeprecationWarning, stacklevel=2)
        return self.daemon

    def setDaemon(self, daemonic):
        """Set whether this thread is a daemon.

        This method is deprecated, use the .daemon property instead.

        """
        import warnings
        warnings.warn('setDaemon() is deprecated, set the daemon attribute instead',
                      DeprecationWarning, stacklevel=2)
        self.daemon = daemonic

    def getName(self):
        """Return a string used for identification purposes only.

        This method is deprecated, use the name attribute instead.

        """
        import warnings
        warnings.warn('getName() is deprecated, get the name attribute instead',
                      DeprecationWarning, stacklevel=2)
        return self.name

    def setName(self, name):
        """Set the name string for this thread.

        This method is deprecated, use the name attribute instead.

        """
        import warnings
        warnings.warn('setName() is deprecated, set the name attribute instead',
                      DeprecationWarning, stacklevel=2)
        self.name = name

class _DummyThread(Thread):

    def __init__(self):
        Thread.__init__(self, name=_newname("Dummy-%d"),
                        daemon=_daemon_threads_allowed())
        self._started.set()
        self._set_ident()
        if _HAVE_THREAD_NATIVE_ID:
            self._set_native_id()
        with _active_limbo_lock:
            _active[self._ident] = self

    def _stop(self):
        pass

    def is_alive(self):
        assert not self._is_stopped and self._started.is_set()
        return True

    def join(self, timeout=None):
        assert False, "cannot join a dummy thread"

# MY CODE

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


"""
print(thread0(que0,sentenceUs))
print(thread1(que1,sentenceUs))
print(thread2(que2,sentenceUs))
print(thread3(que3,sentenceUs))
"""

print("About to create")
t0 = ThreadWithReturnValue(target=thread0.thread0Letters, args=(que0,sentenceUs,))
t1 = ThreadWithReturnValue(target=thread1.thread1Letters, args=(que1,sentenceUs,))
t2 = ThreadWithReturnValue(target=thread2.thread2Letters, args=(que2,sentenceUs,))
t3 = ThreadWithReturnValue(target=thread3.thread3Letters, args=(que3,sentenceUs,))

print("Created!")
print("About to start")

t0.start()
t1.start()
t2.start()
t3.start()

print("Started!")
print("About to join")

thread0Return = t0.join()
thread1Return = t1.join()
thread2Return = t2.join()
thread3Return = t3.join()

print("Joined!")
# os.system("pause")

sentenceUs = thread0Return + thread1Return + thread2Return + thread3Return
sentenceUsRandom = sentenceUs
counter = 1

if sentenceUs == sentenceUser:
    print("Uhh this should be impossible")
elif sentenceUs != sentenceUser:
    print("About to start randomising the generated sentence")
    while sentenceUsRandom != sentenceUser:
        sentenceUsRandomList = list(sentenceUsRandom)
        shuffle(sentenceUsRandomList)
        sentenceUsRandom = ''.join(sentenceUsRandomList)
        print(f"    Random Sentence {counter}: " + sentenceUsRandom)
        counter += 1
    sentenceUs = sentenceUsRandom
    print("Got It!", end=": ")

print(sentenceUs, end=" ")
print("Got it in " + str(counter) + " tries")
end = time()
delta = end-start
if delta < 60:
    print("Took ", delta, "seconds")
elif 60 <= delta <= 3600:
    print("Took ", delta/60, "minutes")
elif delta > 3600:
    print("Took ", (delta/60)/60, "hours")