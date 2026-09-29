# AGENTS.md

本文件是 AI 编码代理（Codex、Claude Code 等）在本仓库工作时的约定。`CLAUDE.md` 只通过 `@AGENTS.md` 引用本文件，请只修改本文件。

## Git 工作流

- 修改前先 `git status -sb`，再 `git pull --ff-only`；有未提交的本地修改导致 pull 失败时，先 stash 或询问用户，pull 成功后再开始修改。
- 修改后按本仓库已有的提交风格 commit，然后 `git push`；push 被拒绝就 `git pull --rebase`，解决冲突后再 push，不要留下未推送的提交。
- 不要提交密钥、`.env`、大文件、构建产物和虚拟环境；它们出现在待提交列表里时先询问。

## 项目说明

- 项目用途、运行方式和目录结构见 `README.md`；确认过的代码约定请补充到本文件。
