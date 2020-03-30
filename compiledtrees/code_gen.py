from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from distutils import sysconfig

import numpy as np
import contextlib
import os
import subprocess
import tempfile
from joblib import Parallel, delayed

import platform

if platform.system() == 'Windows':
    CXX_COMPILER = os.environ['CXX'] if 'CXX' in os.environ else None
    delete_files = False
else:
    CXX_COMPILER = sysconfig.get_config_var('CXX')
    # Set to False to see files in /tmp/
    delete_files = True

EVALUATE_FN_NAME = "evaluate"
ALWAYS_INLINE = "__attribute__((__always_inline__))"


class CodeGenerator(object):
    def __init__(self):
        self._file = tempfile.NamedTemporaryFile(mode='w+b',
                                                 prefix='compiledtrees_',
                                                 suffix='.cpp',
                                                 delete=delete_files)
        self._indent = 0

    @property
    def file(self):
        self._file.flush()
        return self._file

    def write(self, line):
        self._file.write(("  " * self._indent + line + "\n").encode("ascii"))

    @contextlib.contextmanager
    def bracketed(self, preamble, postamble):
        assert self._indent >= 0
        self.write(preamble)
        self._indent += 1
        yield
        self._indent -= 1
        self.write(postamble)


def code_gen_tree(tree, n_classes, evaluate_fn=EVALUATE_FN_NAME, gen=None):
    """
    Generates C code representing the evaluation of a tree.

    Writes code similar to:
    ```
        extern "C" {
          __attribute__((__always_inline__)) double evaluate(float* f) {
            if (f[9] <= 0.175931170583) {
              return 0.0;
            }
            else {
              return 1.0;
            }
          }
        }
    ```

    to the given CodeGenerator object.
    """
    if gen is None:
        gen = CodeGenerator()

    def recur(node):
        if tree.children_left[node] == -1:
            # Add the sum part
            for i in range(0, n_classes):
                # print("Prediction Value is", i, tree.value[node][0][i])
                gen.write(" result[{0}] += (float){1}; ".format(i, (float)(tree.value[node][0][i])/np.sum(tree.value[node][0])))
            # assert tree.value[node].size == 1
            # print("Tree node value is ", i, tree.value[node])
            # gen.write("return {0};".format(1))
            return

        branch = "if (f[{feature}] <= {threshold}f) {{".format(
            feature=tree.feature[node],
            threshold=tree.threshold[node])
        with gen.bracketed(branch, "}"):
            recur(tree.children_left[node])

        with gen.bracketed("else {", "}"):
            recur(tree.children_right[node])

    with gen.bracketed('extern "C" {', "}"):
        fn_decl = "{inline} void {name}(float* f, double *result) {{".format(
            inline=ALWAYS_INLINE,
            name=evaluate_fn)
        with gen.bracketed(fn_decl, "}"):
            recur(0)
    return gen.file


def _gen_tree(i, tree, n_classes):
    """
    Generates cpp code for i'th tree.
    Moved out of code_gen_ensemble scope for parallelization.
    """
    name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
    gen_tree = CodeGenerator()
    return code_gen_tree(tree=tree, n_classes=n_classes, evaluate_fn=name, gen=gen_tree)


def code_gen_ensemble(trees, individual_learner_weight, initial_value,
                      n_classes, gen=None, n_jobs=1):
    """
    Writes code similar to:

    ```
    extern "C" {
      __attribute__((__always_inline__)) double evaluate_partial_0(float* f) {
        if (f[4] <= 0.662200987339) {
          return 1.0;
        }
        else {
          if (f[8] <= 0.804652512074) {
            return 0.0;
          }
          else {
            return 1.0;
          }
        }
      }
    }
    extern "C" {
      __attribute__((__always_inline__)) double evaluate_partial_1(float* f) {
        if (f[4] <= 0.694428026676) {
          return 1.0;
        }
        else {
          if (f[7] <= 0.4402526021) {
            return 1.0;
          }
          else {
            return 0.0;
          }
        }
      }
    }

    extern "C" {
      double evaluate(float* f) {
        double result = 0.0;
        result += evaluate_partial_0(f) * 0.1;
        result += evaluate_partial_1(f) * 0.1;
        return result;
      }
    }
    ```

    to the given CodeGenerator object.
    """
    print ("Max classes is", n_classes)
    if gen is None:
        gen = CodeGenerator()

    tree_files = [_gen_tree(i, tree, n_classes) for i, tree in enumerate(trees)]
    gen.write("#include <stdio.h>")
    with gen.bracketed('extern "C" {', "}"):
        # add dummy definitions if you will compile in parallel
        for i, tree in enumerate(trees):
            name = "{name}_{index}".format(name=EVALUATE_FN_NAME, index=i)
            gen.write("void {name}(float* f, double* result);".format(name=name))

        fn_decl = "double {name}(float* f) {{".format(name=EVALUATE_FN_NAME)
        with gen.bracketed(fn_decl, "}"):
            gen.write("double result[{0}] = {{0.0f}};".format(tree.max_n_classes))
            for i, _ in enumerate(trees):
                increment = "{name}_{index}(f, result);".format(
                    name=EVALUATE_FN_NAME,
                    index=i,)
                gen.write(increment)
            # TODO: Return argmax here
            # gen.write("printf(\"%d %d %d \\n\", result[0], result[1], result[2]);")
            gen.write("double max_value = {0};".format(0))
            gen.write("int max_index = {0};".format(-1))
            loop_decl = "for(int i=0; i<{0}; i++) {{".format(tree.max_n_classes)
            with gen.bracketed(loop_decl, "}"):
                gen.write("result[i]/= {0} + (double)0.0;".format(len(trees)))
                if_decl = "if (max_value < result[i]) {"
                with gen.bracketed(if_decl, "}"):
                    gen.write("max_index = i;")
                    gen.write("max_value = result[i] + (double)0.0;")
                # gen.write(" }")
            # gen.write("printf(\"Max index is %d \\n\", max_index);")
            gen.write("return max_index + (double)0.0;")
    return tree_files + [gen.file]


def _compile(cpp_f):
    if CXX_COMPILER is None:
        raise Exception("CXX compiler was not found. You should set CXX "
                        "environmental variable")
    o_f = tempfile.NamedTemporaryFile(mode='w+b',
                                      prefix='compiledtrees_',
                                      suffix='.o',
                                      delete=delete_files)
    if platform.system() == 'Windows':
        o_f.close()
    _call([CXX_COMPILER, cpp_f, "-c", "-fPIC", "-o", o_f.name, "-O3", "-pipe"])
    return o_f


def _call(args):
    DEVNULL = open(os.devnull, 'w')
    subprocess.check_call(" ".join(args),
                          shell=True, stdout=DEVNULL, stderr=DEVNULL)


def compile_code_to_object(files, n_jobs=1):
    # if ther is a single file then create single element list
    # unicode for filename; name attribute for file-like objects
    if isinstance(files, str) or hasattr(files, 'name'):
        files = [files]

    # Close files on Windows to avoid permission errors
    if platform.system() == 'Windows':
        for f in files:
            f.close()

    o_files = (Parallel(n_jobs=n_jobs, backend='threading')
               (delayed(_compile)(f.name) for f in files))

    so_f = tempfile.NamedTemporaryFile(mode='w+b',
                                       prefix='compiledtrees_',
                                       suffix='.so',
                                       delete=delete_files)
    # Close files on Windows to avoid permission errors
    if platform.system() == 'Windows':
        so_f.close()

    # link trees
    if platform.system() == 'Windows':
        # a hack to overcome large RFs on windows and CMD 9182 chaacters limit
        list_ofiles = tempfile.NamedTemporaryFile(mode='w+b',
                                                  prefix='list_ofiles_',
                                                  delete=delete_files)
        for f in o_files:
            list_ofiles.write((f.name.replace('\\', '\\\\') +
                               "\r").encode('latin1'))
        list_ofiles.close()
        _call([CXX_COMPILER, "-shared", "@%s" % list_ofiles.name, "-fPIC",
               "-flto", "-o", so_f.name, "-O3", "-pipe"])

        # cleanup files
        for f in o_files:
            os.unlink(f.name)
        for f in files:
            os.unlink(f.name)
        os.unlink(list_ofiles.name)
    else:
        _call([CXX_COMPILER, "-shared"] +
              [f.name for f in o_files] +
              ["-fPIC", "-flto", "-o", so_f.name, "-O3", "-pipe"])

    print("so are", so_f)
    return so_f
