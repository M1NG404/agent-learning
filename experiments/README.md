# Fact Extraction Experiment

## 1. 实验目标

验证：

> 给 LLM 一份完整 Markdown 文档，是否能够稳定抽取“主体 + 属性 + 值 + 原文证据”。

当前实验暂不涉及：

* RAG
* Embedding
* Qdrant
* 冲突检测
* Entity Resolution
* Fact Store
* LangExtract

当前数据流：

```text
Markdown
   ↓
添加行号
   ↓
Whole Document
   ↓
LLM
   ↓
Facts JSON
```

---

## 2. Fact 结构

当前 Fact 定义：

```json
{
  "subject_type": "COMPANY",
  "subject_name": "江阴东华铝材科技有限公司",
  "predicate": "贷款本金",
  "value": "11521035.88",
  "unit": "元",
  "effective_date": null,
  "line_start": 25,
  "line_end": 25,
  "evidence": "原文证据"
}
```

核心抽象：

```text
Subject
   ↓
Predicate
   ↓
Value
```

并通过 `evidence + line_start + line_end` 实现原文溯源。

---

## 3. 当前测试文档

第一轮使用一份董事会/股东会决议类 Markdown 文档。

文档中包含：

* 多家公司
* 授信额度
* 抵押/质押担保
* 连带责任保证
* 共同债务人
* 土地使用权
* 房产面积
* “本公司”等跨行指代关系

因此适合用于初步验证复杂主体关系抽取能力。

---

## 4. Prompt v1 结果

### 有效表现

模型能够识别：

* COMPANY
* PROPERTY
* 金额
* 土地面积
* 房产面积
* 日期

模型能够生成基本可用的结构化 Facts。

### 发现的问题

#### 4.1 value 被自动标准化

原文：

```text
贰仟万 元
```

模型输出：

```json
"value": 20000000
```

说明模型在事实抽取过程中同时执行了金额标准化。

这不符合当前实验“忠实抽取原文”的目标。

#### 4.2 跨行主体证据不完整

例如模型识别：

```text
subject_name = 江苏海达科技集团有限公司
```

但对应 evidence 只有：

```text
同意本公司为……
```

公司名称实际上来自前文：

```text
江苏海达科技集团有限公司（以下简称本公司）
```

因此当前 `line_start / line_end` 无法完整证明主体归属。

---

## 5. Prompt v2 调整

新增约束：

1. value 必须保持原文表达。
2. value 必须统一返回字符串。
3. 禁止自动计算或标准化金额。
4. 当主体依赖“本公司”等前文指代时，要求证据覆盖主体定义。

---

## 6. Prompt v2 结果

### 6.1 value 原文保持成功

例如：

```text
贰仟万
壹亿元
20879.3
```

均能够以字符串形式保留。

结论：

> Prompt 可以有效控制 LLM 不进行 value 标准化。

后续如果需要金额标准化，应作为独立步骤处理，例如：

```text
raw_value = "贰仟万"
        ↓
Normalizer
        ↓
normalized_value = 20000000
```

---

### 6.2 复杂关系拆解能力较好

原文中的复杂担保关系可以被拆成多个 Fact，例如：

```text
江阴利泰装饰材料有限公司
    ├── 担保方式 → 抵押/质押担保
    ├── 担保主债权金额 → 贰仟万
    ├── 担保主债权人 → 中国银行股份有限公司江阴支行
    └── 被担保人 → 江阴东华铝材科技有限公司
```

阶段性说明：

> 简单的 `subject + predicate + value` 结构目前仍具有较强表达能力，暂时没有必要立即引入 FactGraph。

---

### 6.3 跨行主体溯源仍未解决

虽然 Prompt 已要求覆盖主体定义行，但模型依然出现：

```text
subject_name = 江苏海达科技集团有限公司
line_start = 42
line_end = 42
evidence = “同意本公司为……”
```

而真正确定：

```text
本公司 = 江苏海达科技集团有限公司
```

需要依赖前面的其他行。

因此当前 schema 可能无法很好表达“非连续证据”。

可能需要进一步验证类似结构：

```json
{
  "subject_evidence": {
    "line_start": 39,
    "line_end": 39
  },
  "fact_evidence": {
    "line_start": 42,
    "line_end": 42
  }
}
```

当前暂不修改 schema，继续通过实验确认是否是普遍问题。

---

## 7. 稳定性问题

同一份文档在不同运行中，抽取出的 Fact 集合存在差异。

例如第一次运行包含：

* 董事会召开日期
* 股东会召开日期

第二次运行中这些事实消失，但新增了：

* 担保方式
* 担保主债权人
* 被担保人
* 法律地位

因此：

> `temperature = 0` 不代表事实集合一定完全稳定。

后续需要重点测试 Recall 稳定性。

---

## 8. 当前阶段结论

目前可以确认：

```text
✅ Whole Document 可以直接进行结构化事实抽取
✅ COMPANY / PROPERTY 等主体基本可以识别
✅ 金额、面积等明确字段抽取效果较好
✅ Prompt 可以约束 value 保持原文
✅ 复杂句可以拆成多个 subject-predicate-value

⚠️ 不同运行之间存在 Fact 集合变化
⚠️ Recall 稳定性仍需要验证
❌ “本公司”等跨行指代的完整证据溯源仍未解决
```

---

## 9. 当前不能得出的结论

现阶段还不能证明：

* Whole Document 一定优于 Chunk
* LLM 可以稳定覆盖所有重要事实
* subject + predicate + value 可以覆盖所有尽调关系
* 当前 Fact schema 可以直接用于生产
* 可以直接基于这些 Facts 做冲突检测

---

## 10. 下一步

下一阶段优先做：

```text
保存每次实验结果
        ↓
扩展到 5 份不同类型 Markdown
        ↓
观察 Precision / Recall / 主体归属 / Evidence
        ↓
判断主要问题属于：
Prompt
Whole Document 长度
Entity Resolution
还是 Fact Schema
```
