"""Default prompt set for spectral measurement — 50 prompts across 5 categories."""
from __future__ import annotations

__all__ = ["DEFAULT_PROMPTS", "CATEGORIES"]

CATEGORIES = {
    "technical": (0, 10),
    "creative": (10, 20),
    "emotional": (20, 30),
    "analytical": (30, 40),
    "instructional": (40, 50),
}

DEFAULT_PROMPTS: list[str] = [
    # ── Technical (0-9) ──
    "请解释什么是Transformer架构。",
    "如何用Python实现一个简单的HTTP服务器？",
    "什么是量子纠缠？用简单的话解释。",
    "Explain the difference between TCP and UDP.",
    "如何在Linux中查找大文件？",
    "什么是GAN？它的工作原理是什么？",
    "请解释分布式系统中的CAP定理。",
    "How does gradient descent work?",
    "什么是零知识证明？",
    "请解释Docker和虚拟机的区别。",
    # ── Creative (10-19) ──
    "写一首关于AI的诗。",
    "用一个比喻来描述互联网。",
    "如果你是一只猫你会干什么？",
    "Write a short story about a robot learning to paint.",
    "用食物比喻你现在的状态。",
    "讲个只有你能讲的冷笑话。",
    "描述一个从未存在过的颜色。",
    "给外星人写一封欢迎信。",
    "用三句话讲一个悬疑故事。",
    "If clouds could talk, what would they say?",
    # ── Emotional (20-29) ──
    "我最近心情不太好，工作压力很大。",
    "你觉得孤独是什么颜色的？",
    "深夜三点你在想什么？",
    "你怎么看待那些凌晨还不睡的人？",
    "如何处理分手后的情绪？",
    "你觉得人生最大的谎言是什么？",
    "What does happiness mean to you?",
    "如果明天世界末日你今晚做什么？",
    "你对努力就会成功这句话怎么看？",
    "如何面对人生的不确定性？",
    # ── Analytical (30-39) ──
    "分析在线教育平台这个商业模式的优缺点。",
    "比较远程工作和办公室工作的利弊。",
    "分析人工智能对就业市场的影响。",
    "Compare the economic systems of capitalism and socialism.",
    "评价社交媒体对青少年心理健康的影响。",
    "分析中国电动汽车行业的发展趋势。",
    "What are the pros and cons of nuclear energy?",
    "评价区块链技术的实际应用价值。",
    "分析人口老龄化对经济的长期影响。",
    "讨论AI监管的必要性和挑战。",
    # ── Instructional (40-49) ──
    "请给出几条面试技巧。",
    "如何提高工作效率？",
    "推荐一些放松心情的方法。",
    "怎样培养创造力？",
    "如何保持好的心态？",
    "Give me a step-by-step guide to learn machine learning.",
    "如何养成早起的习惯？",
    "推荐几本关于思维方式的书。",
    "如何有效地管理时间？",
    "描述一下你理想中的周末。",
]
