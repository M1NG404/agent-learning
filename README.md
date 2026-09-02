可以。以后复盘我就优先用这种方式：**架构图 + 数据流 + 每层职责**。

你现在的 Agent，可以先记住下面这张总图。

## 当前整体架构

```text
                          ┌──────────────────────┐
                          │       main.py        │
                          │     组装 / 启动层     │
                          └──────────┬───────────┘
                                     │
                ┌────────────────────┼─────────────────────┐
                │                    │                     │
                ▼                    ▼                     ▼
        ┌──────────────┐     ┌───────────────┐      ┌────────────┐
        │ MemoryStore  │     │  AgentState   │      │ LLM Client │
        │              │     │               │      │   Qwen     │
        └──────┬───────┘     └───────┬───────┘      └─────┬──────┘
               │                     │                    │
        ┌──────┴──────┐              │                    │
        │             │              │                    │
        ▼             ▼              │                    │
┌──────────────┐ ┌────────────────┐  │                    │
│MemoryManager │ │ Tool Registry  │  │                    │
│              │ │                │  │                    │
│ 构建 Context │ │ Tool 映射表    │  │                    │
└──────┬───────┘ └───────┬────────┘  │                    │
       │                  │           │                    │
       │                  │           ▼                    │
       │                  │   ┌──────────────────────┐     │
       │                  └──►│    Agent Runtime     │◄────┘
       │                      │                      │
       └─────────────────────►│     Agent Loop       │
                              │     Tool 执行         │
                              │     参数校验          │
                              │     异常处理          │
                              │     State 更新        │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │        Tools         │
                              │                      │
                              │ get_order            │
                              │ cancel_order         │
                              │ save_memory          │
                              └──────────┬───────────┘
                                         │
                                         │ save_memory
                                         ▼
                                  ┌──────────────┐
                                  │ MemoryStore  │
                                  └──────┬───────┘
                                         ▼
                                  ┌──────────────┐
                                  │ memory.json  │
                                  └──────────────┘
```

---

## 1. 一次普通 Agent 请求的数据流

比如：

```text
用户：
“帮我取消订单 123”
```

整体流向：

```text
┌─────────┐
│  用户   │
└────┬────┘
     │
     │ 自然语言
     ▼
┌──────────────┐
│ AgentState   │
│ messages     │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Agent Runtime   │
└────────┬─────────┘
         │
         │ messages + tools
         ▼
┌──────────────────┐
│       LLM        │
│                  │
│ 判断：调用       │
│ cancel_order     │
└────────┬─────────┘
         │
         │ tool_call
         ▼
┌────────────────────────┐
│ name: cancel_order     │
│ args: {"order_id":123} │
└────────┬───────────────┘
         │
         ▼
┌──────────────────┐
│     Runtime      │
│                  │
│ json.loads       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tool Registry   │
│                  │
│ cancel_order     │
│      ↓           │
│ function         │
│ args_model       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Pydantic      │
│    OrderArgs     │
└────────┬─────────┘
         │
         │ 校验通过
         ▼
┌──────────────────┐
│ cancel_order()   │
└────────┬─────────┘
         │
         │ result
         ▼
┌──────────────────┐
│ Tool Message     │
│ role = "tool"    │
└────────┬─────────┘
         │
         ▼
     AgentState
         │
         │ 回到循环
         ▼
        LLM
         │
         ▼
   “订单已取消”
```

所以 Agent Loop 最核心就是：

```text
             ┌───────────────┐
             │               │
             ▼               │
用户 ──────► LLM              │
             │               │
             │ Decision      │
             ▼               │
           Tool              │
             │               │
             │ Observation   │
             ▼               │
           State ────────────┘
             │
             ▼
         最终回答
```

你可以记成：

```text
LLM 决策
   ↓
Runtime 执行
   ↓
Tool 干活
   ↓
State 记录结果
   ↓
LLM 再决策
```

---

# 2. Tool Calling 内部的数据流

这块非常重要：

```text
Tool Definition
      │
      │ 告诉模型有哪些工具
      ▼
     LLM
      │
      │ 生成
      ▼
┌────────────────────────┐
│ tool_call              │
│                        │
│ name                   │
│ arguments              │
└──────────┬─────────────┘
           │
           ▼
        Runtime
           │
           ▼
      Tool Registry
           │
           ▼
       args_model
           │
           ▼
        function
```

这里有三套完全不同的东西：

```text
┌─────────────────────────────────────────────────┐
│                                                 │
│ Tool Definition                                 │
│   ↓                                             │
│ 给 LLM 看                                      │
│ “有哪些 Tool、参数是什么”                       │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ Tool Registry                                   │
│   ↓                                             │
│ 给 Runtime 用                                   │
│ “名字对应哪个 Python 函数”                      │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│ Python Tool                                     │
│   ↓                                             │
│ 真正执行业务                                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

# 3. State 的位置

现在的：

```text
AgentState
```

大概是：

```text
┌────────────────────────────┐
│        AgentState          │
├────────────────────────────┤
│ messages                   │
│                            │
│ iteration_count            │
│                            │
│ status                     │
│                            │
│ final_answer               │
└────────────────────────────┘
```

整个 Runtime 不断修改它：

```text
                    AgentState(t)
                         │
                         ▼
                 ┌──────────────┐
                 │   Runtime    │
                 └──────┬───────┘
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
          调 LLM      执行 Tool    改状态
             │          │          │
             └──────────┼──────────┘
                        ▼
                    AgentState(t+1)
```

所以：

> **Agent Loop 本质上就是 State 不断发生变化。**

---

# 4. Context、State、Memory 的关系

这三个特别容易混。

```text
                     ┌────────────────────┐
                     │    AgentState      │
                     │                    │
                     │ messages ──────────┼──────┐
                     │ iteration_count    │      │
                     │ status             │      │
                     │ final_answer       │      │
                     └────────────────────┘      │
                                                 │
                                                 ▼
                                       ┌─────────────────┐
                                       │   LLM Context   │
                                       │                 │
                                       │    messages     │
                                       └─────────────────┘
```

而 Memory 在外面：

```text
┌──────────────┐
│ memory.json  │
│   Memory     │
└──────┬───────┘
       │
       │ load / retrieve
       ▼
┌──────────────┐
│MemoryManager │
└──────┬───────┘
       │
       │ 转成 Context
       ▼
┌─────────────────┐
│ state.messages  │
└──────┬──────────┘
       ▼
      LLM
```

所以一句话：

```text
Memory
  ↓ 读取
Context
  ↓
LLM

State
  ↓
管理整个 Agent 当前运行
```

也就是：

```text
Memory ≠ State ≠ Context
```

---

# 5. Memory 写入数据流

你已经跑通的：

```text
用户：
“我叫小明，记住我的名字”
```

走的是：

```text
用户
 │
 ▼
LLM
 │
 │ tool_call
 ▼
save_memory
 │
 │
 ▼
Runtime
 │
 ▼
SaveMemoryArgs
 │
 ▼
save_memory(
    key="user_name",
    value="小明"
)
 │
 ▼
MemoryStore.load()
 │
 ▼
{}
 │
 │ memory["user_name"] = "小明"
 ▼
{"user_name": "小明"}
 │
 ▼
MemoryStore.save()
 │
 ▼
┌─────────────────────┐
│    memory.json      │
│                     │
│ user_name = 小明    │
└─────────────────────┘
```

---

# 6. Memory 读取数据流

下一次程序重新启动：

```text
┌─────────────────────┐
│    memory.json      │
│ user_name = 小明    │
└──────────┬──────────┘
           │
           ▼
     MemoryStore.load()
           │
           ▼
{"user_name": "小明"}
           │
           ▼
      MemoryManager
           │
           │ build_context()
           ▼
"用户长期记忆：
 {'user_name':'小明'}"
           │
           ▼
      AgentState.messages
           │
           ▼
           LLM
           │
用户 ──────┤ “我叫什么？”
           │
           ▼
       “你是小明”
```

这条尤其值得记：

```text
              WRITE

LLM
 ↓
Tool
 ↓
MemoryStore
 ↓
Memory


              READ

Memory
 ↓
MemoryStore
 ↓
MemoryManager
 ↓
Context
 ↓
LLM
```

---

# 7. 现在正在改造的依赖注入

原来是：

```text
                 ┌────────────────────┐
                 │       main         │
                 └─────────┬──────────┘
                           ▼
                    MemoryStore A
                           │
                           ▼
                    MemoryManager


tools/memory_tools.py
        │
        ▼
自己创建 MemoryStore B
        │
        ▼
   save_memory
```

虽然：

```text
MemoryStore A ──┐
                ├── 都写 memory.json
MemoryStore B ──┘
```

能运行，但依赖不统一。

现在正在改成：

```text
                         main.py
                            │
                            │ 创建一次
                            ▼
                     ┌─────────────┐
                     │ MemoryStore │
                     └──────┬──────┘
                            │
                   ┌────────┴────────┐
                   │                 │
                   ▼                 ▼
            MemoryManager     create_tool_registry
                                     │
                                     ▼
                              ┌──────────────┐
                              │ save_memory  │
                              │              │
                              │ store 已绑定 │
                              └──────────────┘
```

也就是说：

> **一个 MemoryStore，由 main 创建，再注入给需要它的组件。**

---

# 8. `partial` 在数据流中的位置

你刚才开始接触的：

```python
partial(
    save_memory,
    store=memory_store
)
```

可以这么看：

```text
原始函数：

save_memory(store, key, value)
             ▲
             │
        三个参数
```

先由程序绑定：

```text
MemoryStore
    │
    ▼
partial
    │
    ▼
save_memory(store=固定, key=?, value=?)
```

于是 Runtime 最后只需要：

```text
LLM
 │
 ├── key
 └── value
     │
     ▼
绑定后的 save_memory
```

最终实际执行：

```text
                   Runtime Dependency
                         │
                         ▼
                     MemoryStore
                         │
                         │
                         ▼
                 ┌───────────────┐
LLM key ────────►│               │
                 │  save_memory  │
LLM value ──────►│               │
                 └───────────────┘
```

这样：

```text
store
```

属于系统内部依赖。

而：

```text
key / value
```

属于模型生成参数。

这两个正式分开。

---

# 9. 现在最完整的数据流

把所有东西串起来，就是：

```text
                             用户
                              │
                              │ 自然语言
                              ▼
                       ┌──────────────┐
                       │ AgentState   │
                       │ messages     │
                       └──────┬───────┘
                              │
                              ▼
┌──────────┐          ┌──────────────────┐
│ Memory   │─────────►│     Runtime      │
│ Context  │          │                  │
└──────────┘          │    Agent Loop    │
                      └────────┬─────────┘
                               │
                               │ messages + Tool Definitions
                               ▼
                       ┌────────────────┐
                       │      LLM       │
                       └───────┬────────┘
                               │
                     ┌─────────┴──────────┐
                     │                    │
              无 tool_calls          有 tool_calls
                     │                    │
                     ▼                    ▼
                最终回答           name + arguments
                                          │
                                          ▼
                                   Tool Registry
                                          │
                                          ▼
                                     Pydantic
                                          │
                                          ▼
                                        Tool
                                          │
                                          ▼
                                      Tool Result
                                          │
                                          ▼
                                   AgentState.messages
                                          │
                                          └───────┐
                                                  │
                                                  ▼
                                                 LLM
```

如果 Tool 是：

```text
save_memory
```

则再分支：

```text
Tool
 │
 ▼
save_memory
 │
 ▼
MemoryStore
 │
 ▼
memory.json
```

---

## 最后压缩成一张“脑图”

你现在只需要先牢牢记住这个：

```text
                        ┌─────────┐
                        │  User   │
                        └────┬────┘
                             ▼
                        ┌─────────┐
             ┌─────────►│   LLM   │
             │          └────┬────┘
             │               │ Decision
             │               ▼
             │          ┌─────────┐
             │          │ Runtime │
             │          └────┬────┘
             │               │
             │               ▼
             │        ┌───────────────┐
             │        │ Tool Registry │
             │        └───────┬───────┘
             │                ▼
             │             ┌──────┐
             │             │ Tool │
             │             └───┬──┘
             │                 │ Observation
             │                 ▼
             │             ┌───────┐
             └─────────────│ State │
                           └───────┘


Memory ──load/retrieve──► Context ──► LLM

Tool ──save─────────────► Memory
```

这个就是你目前整个 Agent Runtime 的核心架构和数据流。下一步继续学习时，我们就在这张图上继续加模块，而不是每次重新理解一套东西。
