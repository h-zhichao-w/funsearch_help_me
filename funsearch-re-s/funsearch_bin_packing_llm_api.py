import json
import multiprocessing
from typing import Collection, Any
import http.client
from implementation import funsearch
from implementation import config
from implementation import sampler
from implementation import evaluator_accelerate
from implementation import evaluator
from implementation import code_manipulation
import bin_packing_utils
import dataset

import json
import multiprocessing
from typing import Collection, Any
import http.client
from implementation import sampler

import matplotlib.pyplot as plt 
import time

scores_list = []
time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_with_program_path = f'logs/log-24hr-{time_stamp}.txt'

def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions/symbols/'def' declaration before the function body.
    Please see my comments in sampler.LLM (in sampler.py).
    Since the LLM used in this file is not a pure code completion LLM, this trim function is required.

    -Example sample (function & description generated by LLM):
    -------------------------------------
    This is the optimized function ...
    def priority_v2(...) -> ...:
        return ...
    This function aims to ...
    -------------------------------------
    -This function removes the description above the function's signature, and the function's signature.
    -The indent of the code is preserved.
    -Return of this function:
    -------------------------------------
        return ...
    This function aims to ...
    -------------------------------------
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        # find the first 'def' statement in the given code
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class LLMAPI(sampler.LLM):
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        additional_prompt = ('Complete a different and more complex Python function. '
                             'Be creative and you can insert multiple if-else and for-loop in the code logic.'
                             'Only output the Python code, no descriptions.')
        self._additional_prompt = additional_prompt
        self._trim = trim

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content, self._additional_prompt])
        while True:
            try:
                conn = http.client.HTTPSConnection("www.jcapikey.com")
                # conn = http.client.HTTPSConnection("api.chatanywhere.com.cn")
                payload = json.dumps({
                    "max_tokens": 1024,
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                headers = {
                    # 'Authorization': 'Bearer sk-Dtq4Jt0VxwIHMi3QtTcMANiBxuOH0OKZbjKNAao41aRStRz3',
                    'Authorization': 'Bearer sk-CYLL2v3Eu2TxwuQE3807E9293a484024947745D2A213CfD2',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                # trim function
                if self._trim:
                    response = _trim_preface_of_body(response)
                return response
            except Exception as e:      #* 这里如果没有成功通信, 比如 key 输错了, 则会进入死循环
                continue


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code. Implemented by RZ.

    RZ: Sandbox returns the 'score' of the program and:
    1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Args:
            verbose         : Print evaluate information.
            numba_accelerate: Use numba to accelerate the evaluation. It should be noted that not all numpy functions
                              support numba acceleration, such as np.piecewise().
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program: str,
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            inputs: Any,  # refers to the dataset
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            **kwargs  # RZ: add this
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.

        RZ: If the generated code (generated by LLM) is executed successfully,
        the output of this function is the score of a given program.
        RZ: PLEASE NOTE THAT this SandBox is only designed for bin-packing problem.
        """
        dataset = inputs
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)
        if process.is_alive():
            # if the process is not finished in time, we consider the program illegal
            process.terminate()
            process.join()
            results = None, False
        else:
            if not result_queue.empty():
                results = result_queue.get_nowait()     #* 执行到这里, 返回了 None, False !
            else:
                results = None, False

        if self._verbose:
            print(f'================= Evaluated Program =================')
            program_: code_manipulation.Program = code_manipulation.text_to_program(text=program)
            func_to_evolve_: str = kwargs.get('func_to_evolve', 'survey_boot_build')
            function_: code_manipulation.Function = program_.get_function(func_to_evolve_)
            function_: str = str(function_).strip('\n')
            print(f'{function_}')
            print(f'-----------------------------------------------------')
            print(f'Score: {str(results)}')
            print(f'=====================================================')
            print(f'\n\n')

        if results[0] is not None:
            program_: code_manipulation.Program = code_manipulation.text_to_program(text=program)
            func_to_evolve_: str = kwargs.get('func_to_evolve', 'survey_boot_build')
            function_: code_manipulation.Function = program_.get_function(func_to_evolve_)
            function_: str = str(function_).strip('\n')
            log = open(log_with_program_path, 'a')
            log.write(f'================= Evaluated Program =================\n')
            log.write(f'{function_}\n')
            log.write(f'-----------------------------------------------------\n')
            log.write(f'Score: {str(results[0])}\n')
            log.write(f'=====================================================\n')
            log.write(f'\n\n')
            log.close()
            scores_list.append(results[0])
        else:
            scores_list.append(0)

        return results

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_run'
            function_to_run = all_globals_namespace[function_to_run]
            # return the execution results
            results = function_to_run(dataset)
            # results = function_to_run()
            # the results must be int or float
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception as e:     #* 源代码没打印错误信息...
            print("When in Sandbox, there comes an error:", e)        #* 打印出来一看... 艹, 居然是没装 numba 库... 装完后顺利解决
            # if raise any exception, we assume the execution failed
            result_queue.put((None, False))


# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':

    #* 读取 specification.py 文件
    with open('specification.py') as f:
    # with open('/home/jty/Code/zhengkan/deepmind/funsearch_help_me/funsearch-re-s/funsearch-re-s/specification_new_5.py') as f:
        specification2 = f.read()

    inputs = dataset.dataset

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=120)

    # bin_packing_or3 = {'OR3': bin_packing_utils.datasets['OR3']}
    global_max_sample_num = 100  # if it is set to None, funsearch will execute an endless loop
    funsearch.main(
        specification=specification2,
        inputs=inputs,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir='logs/funsearch_llm_api',
    )

    if True:
        # 绘制 scores_list 折线, 并在最高点处画一条水平虚线作为参考线
        plt.plot(scores_list, label='FunSearch')
        plt.axhline(y=(scores_list[0]), color='r', linestyle='--', label='Baseline')
        plt.axhline(y=(max(scores_list)), color='g', linestyle='--', label='Best of all: {0}'.format(
            str(round(max(scores_list), 3))))
        plt.title('Scores of the generated programs')
        plt.xlabel('Sample Number')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(f'logs/scores_list_{time_stamp}.png')
        plt.show()

    

