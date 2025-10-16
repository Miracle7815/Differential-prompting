import generate_variants
import generate_test_input
import execute_test
from openai import OpenAI

BASE_URL = 'https://api.openai-proxy.org/v1'
API_KEY = 'sk-YXeIf5Hzq452SluTP77QPGWOeWHq7GFMqH4C4kwr9uFZhbhv'

client = OpenAI(base_url=BASE_URL , api_key=API_KEY)

model_name = "gpt-4o-mini"

generate_variants.parse_and_generate_variants_for_TrickyBugs(client , model=model_name , k=6 , temperature=0.8)
generate_variants.transform_code_for_TrickyBugs(model_name)

generate_test_input.parse_and_generate_test_for_TrickyBugs(client , model_name)
generate_test_input.extract_test_generator(model_name)
generate_test_input.execute_input_generator(model_name)

execute_test.execute_test_for_TrickyBugs(model_name)
execute_test.calculate_the_coverage(model_name)