import subprocess
import os
import sys
import shutil
import platform
import psutil
import time
import argparse
import json
from llm_helper import is_random_string

# Placeholder for model path. The user should configure this.
# I'll assume a directory structure for the models.
MODEL_DIR = "models-hf/llama-7b" 
CONVERTED_MODEL_NAME = "dragon-model-f16.bin"
# The conversion script for multi-part models will add a suffix. Assuming single part for llama-7b based on config.
# The script logic seems to create consolidated.00.pth -> dragon-model-f16.bin
# and consolidated.01.pth -> dragon-model-f16.bin.1
# Let's assume the main file is without suffix.
CONVERTED_MODEL_PATH = os.path.join(MODEL_DIR, "dragon-model-f16.bin")
EXECUTABLE_NAME = "llama"
EXECUTABLE_PATH = f"./{EXECUTABLE_NAME}" # Will be updated by build_project to be inside the build dir

def print_grading_rules():
    """打印评分规则。"""
    print("="*30)
    print(" " * 10 + "评分标准")
    print("="*30)
    print("步骤 0: 项目构建")
    print("  - 分数: 0")
    print("  - 必须通过才能运行其他测试。")
    print("-" * 30)
    print("步骤 1: 模型转换")
    print("  - 分数: 0")
    print("  - 必须通过才能运行步骤 2 和步骤 3。")
    print("-" * 30)
    print("步骤 2: 模型加载")
    print("  - 分数: 10")
    print("  - 测试转换后的模型能否被无错加载。")
    print("-" * 30)
    print("步骤 3: 推理")
    print("  - 分数: 20")
    print("  - 运行一个简短的推理并检查输出是否有效。")
    print("-" * 30)
    print("步骤 4: 死锁和多线程")
    print("  - 分数: 30")
    print("  - 检查推理过程中是否存在死锁并验证多线程是否启用。")
    print("-" * 30)
    print("步骤 5: 分词器")
    print("  - 分数: 20")
    print("  - 运行 20 个分词器测试用例。")
    print("  - 每个测试用例 1 分，满分 20 分。")
    print("-" * 30)
    print("步骤 6: 结构化输出 (Choice)")
    print("  - 分数: 10")
    print("  - 测试模型是否能从给定选项中选择正确的答案。")
    print("  - 每个测试用例 0.5 分，满分 10 分。")
    print("-" * 30)
    print("步骤 7: 结构化输出 (JSON)")
    print("  - 分数: 20")
    print("  - 测试模型是否能根据 schema 生成有效的 JSON。")
    print("  - 每个测试用例 1 分，满分 20 分。")
    print("="*30)
    print("总分: 110 分")
    print("="*30)

def test_step_0_build_project():
    """Builds the C++ project using cmake."""
    print("="*20)
    print("Building project with CMake...")
    build_dir = "build"
    try:
        if not os.path.exists("CMakeLists.txt"):
            print("CMakeLists.txt not found. Cannot build project.")
            return False

        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        
        cmake_cmd = ["cmake", ".."]
        subprocess.run(cmake_cmd, cwd=build_dir, check=True, capture_output=True, text=True)
        
        make_cmd = ["make", "-j", "18"]
        subprocess.run(make_cmd, cwd=build_dir, check=True, capture_output=True, text=True)

        exe_path = os.path.join(build_dir, EXECUTABLE_NAME)
        if os.path.exists(exe_path):
            print(f"Build successful. Executable at: {exe_path}")
            globals()["EXECUTABLE_PATH"] = exe_path
            return True
        else:
            print(f"Build failed, executable not found at {exe_path}.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Error during build:")
        print(e.stdout)
        print(e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"Error: '{e.filename}' command not found. Please ensure it's installed and in your PATH.")
        return False

def test_step_1_model_conversion():
    """
    Tests step 1: Model conversion using convert-pth-to-dragon.py.
    """
    print("="*20)
    print("Step 1: Testing model conversion...")
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory not found at {MODEL_DIR}")
        print(f"Creating dummy model directory and files for testing...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "params.json"), "w") as f:
            f.write('{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-06, "vocab_size": 32000}')
        # create dummy tokenizer.model
        with open(os.path.join(MODEL_DIR, "tokenizer.model"), "w") as f:
            # SentencePieceProcessor needs a valid model file, an empty file will cause errors.
            # We can't easily create a valid one, so we'll have to assume this step might be fragile
            # without real model files. For now, this is a placeholder.
            pass
        # create dummy consolidated.00.pth
        with open(os.path.join(MODEL_DIR, "consolidated.00.pth"), "w") as f:
            f.write("")
        
    try:
        # ftype=1 for float16
        cmd = ["python", "convert-pth-to-dragon.py", MODEL_DIR, "1"]
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Model conversion script ran successfully.")
        
        # The script may create multiple parts. Let's check for the first part.
        base_output_path = os.path.join(MODEL_DIR, f"dragon-model-f16.bin")
        if os.path.exists(base_output_path) or os.path.exists(base_output_path + ".0"):
             print(f"Converted model found.")
             # The script might output multiple files, we'll use the base name for the next steps
             # and assume the C++ code knows how to find the other parts.
             # The convert script creates dragon-model-f16.bin (for part 0).
             globals()["CONVERTED_MODEL_PATH"] = base_output_path
             return True
        else:
            print(f"Error: Converted model not found after running conversion script.")
            print(process.stdout)
            print(process.stderr)
            return False

    except subprocess.CalledProcessError as e:
        print("Error during model conversion:")
        print(e.stdout)
        print(e.stderr)
        return False

def test_step_2_model_loading():
    """
    Tests step 2: Loading the converted model.
    Checks for magic number error or memory access errors.
    """
    print("="*20)
    print("Step 2: Testing model loading...")
    
    model_loader_path = os.path.join("build", "model-loader")

    if not os.path.exists(model_loader_path):
        print(f"Error: model-loader executable not found at '{model_loader_path}'. Build might have failed.")
        return False
        
    cmd = [model_loader_path, CONVERTED_MODEL_PATH]
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        print("Model loading test passed (no crash or magic number error).")
        return True
    except subprocess.TimeoutExpired:
        print("Error: Model loading timed out. Possible deadlock.")
        return False
    except subprocess.CalledProcessError as e:
        print("Error: The program crashed during model loading. This might be the memory bug or a magic number error.")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def test_step_3_inference():
    """
    Tests step 3: Full inference run.
    Checks for meaningless output.
    """
    print("="*20)
    print("Step 3: Testing inference for deadlock and output quality...")
    prompt = " Once upon a time"
    cmd = [EXECUTABLE_PATH, "-m", CONVERTED_MODEL_PATH, "-n", "20"]
    try:
        # 1-minute timeout for deadlock detection
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        # The C++ program prints the prompt first, so we should strip it.
        output = process.stdout.replace(prompt, "").strip()
        print(f"Inference output: {output}")

        if is_random_string(output):
            print("Test failed: Output is not meaningful.")
            return False
        
        print("Inference test passed.")
        return True

    except subprocess.TimeoutExpired:
        print("Error: Inference timed out after 1 minutes. A deadlock is likely.")
        return False
    except subprocess.CalledProcessError as e:
        print("Error: The program crashed during inference.")
        print(e.stdout)
        print(e.stderr)
        return False
    
def test_step_4_deadlock():
    """
    Tests step 4: Deadlock detection and thread count.
    Checks for deadlocks and ensures more than one thread is used.
    """
    print("="*20)
    print("Step 4: Testing deadlock detection and thread count...")
    prompt = " Once upon a time"
    # 12 threads requested, we expect more than 2 to be used.
    cmd = [EXECUTABLE_PATH, "-m", CONVERTED_MODEL_PATH, "-n", "20", "-t", "6"]
    timeout_seconds = 120
    max_threads = 0

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        p = psutil.Process(process.pid)
        
        start_time = time.time()
        # Monitor the process while it's running and within the timeout window
        while p.is_running() and (time.time() - start_time) < timeout_seconds:
            try:
                num_threads = p.num_threads()
                if num_threads > max_threads:
                    max_threads = num_threads
            except psutil.NoSuchProcess:
                # Process finished between checks
                break
            time.sleep(0.05)

        print(f"Max threads observed during execution: {max_threads}")

        # Check if the process is still running after the timeout period
        if process.poll() is None:
            print(f"Error: Inference timed out after {timeout_seconds} seconds. A deadlock is likely.")
            p.kill()
            process.wait()
            return False
        
        # Check if multithreading was actually used
        if max_threads <= 2:
            print(f"Test failed: The program ran with only {max_threads} thread(s), but was expected to use more for the '-t 12' parameter.")
            return False

        print("Deadlock test passed. No deadlock detected and multithreading confirmed.")
        return True

    except psutil.NoSuchProcess:
        # This can happen if the process finishes extremely quickly.
        print("Process finished too quickly to monitor threads. Checking output...")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print("Error: The program crashed very quickly.")
            print(f"Stderr: {stderr}")
            return False
        output = stdout.replace(prompt, "").strip()
        if is_random_string(output):
            print("Test failed: Output is not meaningful.")
            return False
        print("Inference test passed (process was too fast for thread monitoring).")
        return True
    except FileNotFoundError:
        print(f"Error: Executable not found at {EXECUTABLE_PATH}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Ensure the subprocess is terminated if it's still running
        if 'process' in locals() and process.poll() is None:
            process.kill()
        return False

def test_step_5_tokenizer():
    """
    Tests step 5: Tokenizer implementation using the dedicated 'tokenizer' executable.
    """
    print("="*20)
    print("Step 5: Testing tokenizer...")

    tokenizer_exe_path = os.path.join("build", "tokenizer")

    if not os.path.exists(tokenizer_exe_path):
        print(f"Error: Tokenizer executable not found at '{tokenizer_exe_path}'. Build might have failed or is incomplete.")
        return 0

    if not os.path.exists(CONVERTED_MODEL_PATH):
        print(f"Error: Converted model '{CONVERTED_MODEL_PATH}' not found. Step 1 (model conversion) must run successfully first.")
        return 0

    prompts = [
        "All animals are equal, but some animals are more equal than others.",

        "Life is like a box of chocolates. You never know what you're gonna get.",

        "We hold these truths to be self-evident, that all men are created equal"
        "that they are endowed by their Creator with certain unalienable Rights, "
        "that among these are Life, Liberty and the pursuit of Happiness.",

        "I disapprove of what you say, but I will defend to the death your right to say it.",

        "Man is born free, and everywhere he is in chains.",

        "也许每一个男子全都有过这样的两个女人，至少两个。娶了红玫瑰，久而久之，红的变成了墙上的一抹蚊子血，"
        "白的还是床前明月光；娶了白玫瑰，白的便是衣服上沾的一粒饭黏子，红的却是心口上一颗朱砂痣。",

        # ..........
        # a lot of hidden test cases
        # ..........
    ]

    expected_outputs = [
        "1 2178 15006 526 5186 29892 541 777 15006 526 901 5186 1135 4045 29889",
        "1 4634 338 763 263 3800 310 3060 29883 23167 29879 29889 887 2360 1073 825 366 29915 276 330 11586 679 29889",
        "1 1334 4808 1438 8760 29879 304 367 1583 29899 5750 1693 29892 393 599 1757 526 2825 5186 5747 896 526 1095 20937 491 1009 6760 1061 411 3058 1185 492 12007 26863 29892 393 4249 1438 526 4634 29892 25421 29891 322 278 12359 3121 310 379 932 3335 29889",
        "1 306 8796 29878 994 310 825 366 1827 29892 541 306 674 24663 304 278 4892 596 1492 304 1827 372 29889",
        "1 2315 338 6345 3889 29892 322 16978 540 338 297 9704 29879 29889",

        "1 29871 30953 235 177 187 31951 30287 30502 31203 30319 30753 30769 30417 31138 30810 31819 30210 31977 30502 30647 30313 30214 235 138 182 31022 31977 30502 30267 232 171 185 30743 31869 234 145 174 234 148 179 30214 31347 31325 31347 30577 30214 31869 30210 31462 30494 30743 232 165 156 30429 30210 30287 233 141 188 235 157 141 30319 235 164 131 30214 30868 30210 31994 30392 232 189 141 30658 30592 30534 30867 31608 232 171 185 30743 30868 234 145 174 234 148 179 30214 30868 30210 231 193 194 30392 235 164 166 31520 30429 233 181 193 30210 30287 234 181 149 236 168 176 236 190 146 30319 30214 31869 30210 232 144 183 30392 30869 30856 30429 30287 236 165 154 31759 234 163 133 234 154 166 30267"
        # ..........
        # a lot of hidden test cases
        # ..........
    ]

    # read test-cases/test-case-of-tokinzer.txt
    if os.path.exists("test-cases/test-case-of-tokinzer.txt"):
        with open("test-cases/test-case-of-tokinzer.txt", "r") as f:
            prompts = f.readlines()
    # read test-cases/test-case-of-tokinzer-expected-output.txt
    if os.path.exists("test-cases/test-case-of-tokinzer-expected-output.txt"):
        with open("test-cases/test-case-of-tokinzer-expected-output.txt", "r") as f:
            expected_outputs = f.readlines()

    passed_count = 0
    total_tests = len(prompts)
    # From tokenizer.cpp, the command is <executable> <model_path> <prompt>
    # Note: tokenizer.cpp adds a leading space to the prompt before tokenizing.
    for i, (prompt, expected_output) in enumerate(zip(prompts, expected_outputs)):
        run_cmd = [tokenizer_exe_path, CONVERTED_MODEL_PATH, prompt]

        print(f"Running tokenizer test {i+1}/{total_tests}: ...")
        try:
            process = subprocess.run(run_cmd, check=True, capture_output=True, text=True, timeout=60)
            output = process.stdout.strip()

            if output == expected_output:
                passed_count += 1
            else:
                print(f"Tokenizer test {i+1} FAILED. Expected '{expected_output}', got '{output}'")

        except subprocess.TimeoutExpired:
            print(f"Error: Tokenizer test {i+1} timed out.")
        except subprocess.CalledProcessError as e:
            print(f"Tokenizer test {i+1} program crashed. The function might be implemented incorrectly.")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
    
    if passed_count == total_tests:
        print("Tokenizer test fully passed.")
    else:
        print(f"Tokenizer test partially passed: {passed_count}/{total_tests} cases passed.")
        
    return passed_count


def test_step_6_structure_output_choice():
    """
    Tests step 6: Structured output (choice).
    """
    print("="*20)
    print("Step 6: Testing structured output (choice)...")
    
    test_cases = [
        {
            "prompt": "Which city is the capital of China?",
            "options": ["Beijing", "Shanghai", "Hongkong", "Tokyo"],
            "expected": "Beijing"
        },
        {
            "prompt": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "expected": "4"
        },
        {
            "prompt": "What is the opposite of hot?",
            "options": ["warm", "cold", "tepid", "burning"],
            "expected": "cold"
        }
        # ..........
        # a lot of hidden test cases
        # ..........
    ]

    passed_count = 0
    total_cases = len(test_cases)
    for i, case in enumerate(test_cases):
        prompt = case["prompt"]
        options_str = ",".join(case["options"])
        expected = case["expected"]
        
        cmd = [
            EXECUTABLE_PATH, 
            "-m", CONVERTED_MODEL_PATH, 
            "-t", "1", # single thread, avoid deadlock
            "-p", prompt,
            "--structure-output-choice",
            "--structure-output-choice-option", options_str,
            "-n", "10" 
        ]

        print(f"Running choice test {i+1}/{total_cases}...")
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            output = process.stdout.replace(prompt, "").strip()
            print(f"Model output: '{output}'")

            if output == expected:
                passed_count += 1
            else:
                print(f"Test failed. Prompt: '{prompt}'. Expected '{expected}', but got '{output}'")

        except subprocess.TimeoutExpired:
            print(f"Error: Test {i+1} timed out. A deadlock is likely.")
        except subprocess.CalledProcessError as e:
            print(f"Error: The program crashed during test {i+1}.")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            
    if passed_count == total_cases:
        print("Structured output (choice) test fully passed.")
    else:
        print(f"Structured output (choice) test partially passed: {passed_count}/{total_cases} cases passed.")
        
    return passed_count

def test_step_7_structure_output_json():
    """
    Tests step 7: Structured output (JSON).
    """
    print("="*20)
    print("Step 7: Testing structured output (JSON)...")

    test_cases = [
        {
            "prompt": "Provide user details for John Doe, who is 30 years old and lives in New York.",
            "schema": "name:string,age:number,city:string",
            "validate": lambda d: (
                isinstance(d.get("name"), str) and "john doe" in d.get("name", "").lower() and
                isinstance(d.get("age"), (int, float)) and d.get("age") == 30 and
                isinstance(d.get("city"), str) and "new york" in d.get("city", "").lower()
            )
        },
        {
            "prompt": "Create a JSON for a product with name 'Laptop', ID 12345, and availability true.",
            "schema": "product_name:string,product_id:number,in_stock:boolean",
            "validate": lambda d: (
                isinstance(d.get("product_name"), str) and "laptop" in d.get("product_name", "").lower() and
                isinstance(d.get("product_id"), int) and d.get("product_id") == 12345 and
                isinstance(d.get("in_stock"), bool) and d.get("in_stock") is True
            )
        },
        {
            "prompt": "Summarize the weather: temperature is 25.3 C, humidity is 60%, and it is sunny.",
            "schema": "temp:number,humidity:number,conditions:string",
             "validate": lambda d: (
                isinstance(d.get("temp"), (int, float)) and 25 <= d.get("temp", 0) <= 26 and
                isinstance(d.get("humidity"), int) and d.get("humidity") == 60 and
                isinstance(d.get("conditions"), str) and "sunny" in d.get("conditions", "").lower()
            )
        }
        # ..........
        # a lot of hidden test cases
        # ..........
    ]

    passed_count = 0
    total_cases = len(test_cases)
    for i, case in enumerate(test_cases):
        prompt = case["prompt"]
        schema_str = case["schema"]
        
        cmd = [
            EXECUTABLE_PATH,
            "-m", CONVERTED_MODEL_PATH,
            "-t", "1",
            "-p", prompt,
            "--structure-output-json",
            "--structure-output-json-key", schema_str,
            "-n", "100"
        ]

        print(f"Running JSON test {i+1}/{total_cases}...")
        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            output = process.stdout.replace(prompt, "").strip()
            print(f"Model output: '{output}'")
            
            json_output = None
            try:
                json_output = json.loads(output)
            except json.JSONDecodeError:
                print(f"Test failed. Output is not valid JSON: {output}")
                continue

            if case["validate"](json_output):
                passed_count += 1
            else:
                print(f"Test failed. JSON output does not match expected structure or values: {json_output}")

        except subprocess.TimeoutExpired:
            print(f"Error: Test {i+1} timed out. A deadlock is likely.")
        except subprocess.CalledProcessError as e:
            print(f"Error: The program crashed during test {i+1}.")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")

    if passed_count == total_cases:
        print("Structured output (JSON) test fully passed.")
    else:
        print(f"Structured output (JSON) test partially passed: {passed_count}/{total_cases} cases passed.")

    return passed_count


def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Run grading tests for the dragon.cpp project.")
    parser.add_argument('--rules', action='store_true', help='Display the grading criteria and exit.')
    args = parser.parse_args()

    if args.rules:
        print_grading_rules()
        sys.exit(0)

    scores = {
        "step_1_model_conversion": 0,
        "step_2_model_loading": 0,
        "step_3_inference": 0,
        "step_4_deadlock": 0,
        "step_5_tokenizer": 0,
        "step_6_structure_output_choice": 0,
        "step_7_structure_output_json": 0,
    }
    
    if not test_step_0_build_project():
        print("Aborting tests due to build failure.")
        sys.exit(1)

    # Step 1: Model Conversion (0 points)
    if test_step_1_model_conversion():
        scores["step_1_model_conversion"] = 0 # Passed but 0 points
        
        # Step 2: Model Loading (10 points)
        if test_step_2_model_loading():
            scores["step_2_model_loading"] = 10
        
        # Step 3: Inference (20 points)
        if test_step_3_inference():
            scores["step_3_inference"] = 20

        # These tests can run if the base model works
        # Step 6: Structure Output Choice Test (10 points)
        scores["step_6_structure_output_choice"] = test_step_6_structure_output_choice() * 0.5
        
        # Step 7: Structure Output JSON Test (20 points)
        scores["step_7_structure_output_json"] = test_step_7_structure_output_json()
    else:
        print("Skipping model loading and inference tests as model conversion failed.")

    # Step 4: Deadlock test (30 points)
    if test_step_4_deadlock():
        scores["step_4_deadlock"] = 30
        
    # Step 5: Tokenizer test (20 points total)
    tokenizer_passed_count = test_step_5_tokenizer()
    if tokenizer_passed_count > 0:
        # 20 points total.
        scores["step_5_tokenizer"] = round(tokenizer_passed_count)

    print("\n" + "="*20)
    print("       GRADING SUMMARY")
    print("="*20)
    total_score = 0
    for test, score in scores.items():
        status = "PASSED" if score > 0 else "FAILED"
        if test == "step_1_model_conversion" and "step_1_model_conversion" in scores: # Special case for 0-point pass
            status = "PASSED"
        
        point_str = "point" if score == 1 else "points"
        if test == "step_5_tokenizer":
             max_points = 20
             print(f"{test}: {score} / {max_points} {point_str}")
        elif test == "step_1_model_conversion":
            print(f"{test}: {status} (0 points)")
        else:
            max_points_map = {
                "step_2_model_loading": 10,
                "step_3_inference": 20,
                "step_4_deadlock": 30,
                "step_6_structure_output_choice": 10,
                "step_7_structure_output_json": 20,
            }
            max_points = max_points_map.get(test, 0)
            print(f"{test}: {score} / {max_points} points")

        total_score += score
    
    print("-" * 20)
    total_possible_score = 10 + 20 + 30 + 20 + 10 + 20
    print(f"Overall score: {total_score} / {total_possible_score}")

if __name__ == "__main__":
    main() 