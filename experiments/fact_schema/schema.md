# Fact Schema v0.1

## Attribute Fact

用于描述实体自身属性。

字段：

- fact_type
- subject_type
- subject
- predicate
- value
- unit
- evidence
- line_start

## Relation Fact

用于描述多个实体之间的业务关系。

字段：

- fact_type
- relation_type
- participants
- attributes
- evidence
- line_start

## Subject Type

- COMPANY
- PERSON
- PROPERTY
- CLAIM
- CONTRACT
- CASE
- OTHER

## Predicate

第一阶段：

- MEETING_DATE
- MEETING_LOCATION
- AREA
- CERTIFICATE_NO

## Relation Type

第一阶段：

- GUARANTEE
- DEBT
- OWNERSHIP