# Entity Resolution Experiment

## 1. 实验目标

验证尽调文档中的指代表达能否稳定解析到正确实体。

重点测试：

```text
“本公司”
    ↓
具体是哪家公司？
```

---

## 2. 实验方法

使用同一模型分别测试两种上下文：

### Local Context

只保留三个主体及其对应的“本公司”引用。

### Whole Document

使用完整董事会 / 股东会决议文档，其中存在多个公司和重复模板。

---

## 3. Local Context 结果

三个“本公司”均解析正确：

```text
本公司
→ 江苏海达科技集团有限公司

本公司
→ 江阴海达特种人革有限公司

本公司
→ 江阴海达彩涂有限公司
```

结果：

```text
3 / 3 正确
```

---

## 4. Whole Document 结果

完整文档中四个“本公司”均解析正确：

```text
本公司
→ 江阴利泰装饰材料有限公司

本公司
→ 江苏海达科技集团有限公司

本公司
→ 江阴海达特种人革有限公司

本公司
→ 江阴海达彩涂有限公司
```

结果：

```text
4 / 4 正确
```

---

## 5. 实验结论

当前结果表明：

```text
✅ Local Context 可以正确完成 Entity Resolution

✅ Whole Document 也可以正确完成 Entity Resolution
```

因此目前没有证据证明：

> 长文档本身会导致“本公司”实体指代失败。

此前 LangExtract 实验中出现的主体漂移，更可能来自多个任务同时耦合：

```text
事实发现
+
Extraction 生成
+
Predicate 生成
+
Value 提取
+
Entity Resolution
+
结构化输出
```

而不是模型单独缺乏 Entity Resolution 能力。

---

## 6. 当前推论

相比让一次 LLM 调用同时完成全部任务，更值得验证两阶段方案：

```text
Markdown
   ↓
Entity Resolution
   ↓
mention → entity

Markdown
   ↓
LangExtract
   ↓
Grounded Extraction

两者结合
   ↓
Final Fact
```

例如：

```text
原文：
“同意本公司为……提供连带保证责任担保”
```

Entity Resolution：

```text
本公司
→ 江苏海达科技集团有限公司
```

LangExtract：

```text
predicate = 担保方式
value = 连带保证责任担保
source_span = 原文位置
```

最终组合：

```text
江苏海达科技集团有限公司
    ↓
担保方式
    ↓
连带保证责任担保
```

---

## 7. 下一步

实验 3：

> Entity Resolution + LangExtract 两阶段组合。

目标：

验证先独立解决实体指代，再进行 Grounded Fact Extraction，是否能够避免 LangExtract 单阶段抽取时出现的主体漂移。
