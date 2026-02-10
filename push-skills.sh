#!/bin/bash

SkillTop=~/.agents/skills
ls -l $SkillTop

for xx in */SKILL.md; do
  SrcDir=$(dirname $xx)
  SkillName=$(basename $SrcDir)
  echo "--------------------"
  echo $SkillName
  rsync -av --delete \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude '.venv/' \
    --exclude '*.pyc' \
    $SkillName/ $SkillTop/$SkillName/
done
