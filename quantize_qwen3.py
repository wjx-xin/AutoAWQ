import time
import json
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
quant_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B-AWQ"
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

print(f"Loading model from {model_path}...")
t0 = time.time()
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    safetensors=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Model loaded in {time.time() - t0:.1f}s")

# Build local calibration data since HuggingFace is unreachable.
# Use WikiText-like passages in English and Chinese (Qwen3 is bilingual).
calib_texts = [
    # English passages
    "The history of artificial intelligence began in antiquity, with myths, stories, and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen.",
    "Quantum computing is a type of computation that harnesses the collective properties of quantum states, such as superposition, interference, and entanglement, to perform calculations.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
    "The Python programming language is widely used in data science, machine learning, web development, and automation.",
    "Transformers are a type of neural network architecture that has revolutionized natural language processing and computer vision.",
    "Large language models are trained on massive text corpora and can perform a wide variety of tasks including translation, summarization, and question answering.",
    "A GPU accelerates computation by performing many operations in parallel, making it ideal for training neural networks.",
    "The attention mechanism allows neural networks to focus on relevant parts of the input when producing each part of the output.",
    "Reinforcement learning is an area of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.",
    "Computer vision enables machines to interpret and understand visual information from the world, such as images and videos.",
    "Data preprocessing is a crucial step in the machine learning pipeline that involves cleaning and transforming raw data into a format suitable for analysis.",
    "Transfer learning allows models trained on one task to be fine-tuned for another related task, significantly reducing the amount of data needed.",
    "The Internet of Things (IoT) refers to the network of physical objects embedded with sensors and software that connect and exchange data over the internet.",
    "Cloud computing provides on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user.",
    "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions across a network of computers.",
    "Cybersecurity involves protecting computer systems and networks from theft, damage, or disruption of their hardware, software, or electronic data.",
    "Edge computing is a distributed computing paradigm that brings computation and data storage closer to the sources of data to improve response times and save bandwidth.",
    "Robotics combines mechanical engineering, electrical engineering, and computer science to design, construct, and operate machines that can assist or replace human actions.",
    # Chinese passages
    "人工智能的发展历史可以追溯到古代，人类一直梦想着创造能够思考和行动的机器。从古希腊的青铜巨人到现代的深度学习模型，这个领域经历了漫长而曲折的发展过程。",
    "量子计算利用量子力学的基本原理，如叠加态和纠缠态，来进行计算。与传统计算机相比，量子计算机在某些特定问题上具有指数级的加速优势。",
    "深度学习是机器学习的一个重要分支，它通过构建多层神经网络来学习数据的层次化表示。近年来，深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。",
    "自然语言处理是人工智能领域的一个重要研究方向，它致力于让计算机能够理解、解释和生成人类语言。随着大规模预训练语言模型的出现，自然语言处理技术取得了巨大的进步。",
    "大语言模型如GPT和BERT等，通过在海量文本数据上进行预训练，获得了强大的语言理解和生成能力。这些模型在各种自然语言处理任务上都表现出了卓越的性能。",
    "注意力机制是现代神经网络架构中的一种关键技术，它允许模型在处理输入序列时动态地关注不同位置的信息，从而有效地捕捉长距离依赖关系。",
    "强化学习是一种通过与环境交互来学习最优策略的机器学习方法。在强化学习中，智能体通过尝试不同的动作并根据环境反馈的奖励信号来不断优化其行为策略。",
    "计算机视觉技术使机器能够理解和分析图像和视频中的视觉信息。从人脸识别到自动驾驶，计算机视觉技术在众多领域都有着广泛的应用。",
    "数据科学是一个跨学科领域，它结合了统计学、计算机科学和领域知识，旨在从数据中提取有价值的见解和知识。",
    "开源软件运动提倡软件的源代码应该公开可用，允许任何人查看、修改和分发。这种开放协作的模式极大地推动了软件技术的发展。",
]

print(f"Using {len(calib_texts)} calibration texts (local, no internet needed)")

print(f"\nStarting quantization...")
print(f"Config: {quant_config}")
t0 = time.time()
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_texts,
    max_calib_samples=128,
    max_calib_seq_len=512,
)
print(f"Quantization completed in {time.time() - t0:.1f}s")

print(f"\nSaving quantized model to {quant_path}...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Model saved successfully to {quant_path}")
