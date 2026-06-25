"""Tests for SkillManager — structured metadata, utility tracking, forgetting curve, pruning, relevance selection."""
import os, sys, uuid
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.skill_manager import SkillManager


def test_load_skills_with_metadata(tmp_path):
    skills_file = tmp_path / "skills.md"
    skills_file.write_text(
        '<!-- skill_id: abc123 -->\n'
        '<!-- utility_score: 0.8 -->\n'
        '<!-- usage_count: 5 -->\n'
        '<!-- role: coding -->\n'
        'Test skill text\n'
        '---\n'
    )
    mgr = SkillManager(str(skills_file))
    skills = mgr.load()
    assert len(skills) == 1
    assert skills[0]['skill_id'] == 'abc123'
    assert skills[0]['utility_score'] == 0.8
    assert skills[0]['usage_count'] == 5
    assert skills[0]['role'] == 'coding'
    assert skills[0]['text'] == 'Test skill text'
    print("  PASS: load skills with structured metadata")


def test_load_plain_text_fallback(tmp_path):
    skills_file = tmp_path / "skills.md"
    skills_file.write_text("- Ao executar X, faca Y.\n- Ao executar Z, faca W.\n")
    mgr = SkillManager(str(skills_file))
    skills = mgr.load()
    assert len(skills) == 2
    for s in skills:
        assert 'skill_id' in s
        assert s['utility_score'] == 0.5
    print("  PASS: plain text fallback with UUIDs")


def test_save_and_reload(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    skills = [{
        'skill_id': 'test-1',
        'extracted_at': '2026-06-25T00:00:00+00:00',
        'role': 'general',
        'usage_count': 3,
        'success_count': 2,
        'failure_count': 1,
        'last_used': '2026-06-25T00:00:00+00:00',
        'utility_score': 0.6,
        'text': '- Test skill',
    }]
    mgr.save(skills)
    reloaded = mgr.load()
    assert len(reloaded) == 1
    assert reloaded[0]['skill_id'] == 'test-1'
    assert reloaded[0]['utility_score'] == 0.6
    assert reloaded[0]['usage_count'] == 3
    print("  PASS: save and reload preserves metadata")


def test_bayesian_update(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    mgr.save([{
        'skill_id': 'abc',
        'utility_score': 0.5,
        'success_count': 0,
        'failure_count': 0,
        'usage_count': 0,
        'role': 'general',
        'text': 'test',
    }])
    mgr.update_utility('abc', 'success')
    skills = mgr.load()
    s = skills[0]
    assert s['usage_count'] == 1
    assert s['success_count'] == 1
    assert abs(s['utility_score'] - 2/3) < 0.01  # 0.6667 vs 0.666...
    mgr.update_utility('abc', 'success')
    mgr.update_utility('abc', 'failure')
    skills = mgr.load()
    s = skills[0]
    assert s['usage_count'] == 3
    assert s['success_count'] == 2
    assert s['failure_count'] == 1
    assert abs(s['utility_score'] - 3/5) < 0.01  # 0.6
    print(f"  PASS: Bayesian update -> {s['utility_score']}")


def test_forgetting_curve(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    skills = [{
        'skill_id': 'old',
        'utility_score': 0.9,
        'last_used': '2025-01-01T00:00:00+00:00',  # ~175 days ago
        'role': 'general',
        'text': 'old skill',
    }, {
        'skill_id': 'recent',
        'utility_score': 0.9,
        'last_used': '2026-06-24T00:00:00+00:00',  # ~1 day ago
        'role': 'general',
        'text': 'recent skill',
    }]
    decayed = mgr._apply_forgetting_curve(skills)
    old = [s for s in decayed if s['skill_id'] == 'old'][0]
    recent = [s for s in decayed if s['skill_id'] == 'recent'][0]
    assert old['utility_score'] < 0.01, f"Old skill should decay heavily: {old['utility_score']}"
    assert recent['utility_score'] > 0.8, f"Recent skill should retain: {recent['utility_score']}"
    print(f"  PASS: forgetting curve (old={old['utility_score']:.4f}, recent={recent['utility_score']:.4f})")


def test_pruning(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    skills = [
        {'skill_id': 'good', 'utility_score': 0.8, 'usage_count': 15, 'text': 'good'},
        {'skill_id': 'bad', 'utility_score': 0.2, 'usage_count': 15, 'text': 'bad'},
        {'skill_id': 'new', 'utility_score': 0.3, 'usage_count': 3, 'text': 'new'},
    ]
    pruned = mgr.prune(skills)
    assert len(pruned) == 2
    assert 'good' in [s['skill_id'] for s in pruned]
    assert 'bad' not in [s['skill_id'] for s in pruned]
    assert 'new' in [s['skill_id'] for s in pruned]  # usage_count 3 < 10 threshold
    print("  PASS: pruning removes low-utility high-usage skills")


def test_selection_by_relevance(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    skills = [
        {'skill_id': 'a', 'utility_score': 0.9, 'role': 'general',  'last_used': '2026-06-24T00:00:00Z', 'usage_count': 5, 'text': 'a' * 50},
        {'skill_id': 'b', 'utility_score': 0.5, 'role': 'security', 'last_used': '2026-06-24T00:00:00Z', 'usage_count': 3, 'text': 'b' * 50},
        {'skill_id': 'c', 'utility_score': 0.7, 'role': 'general',  'last_used': '2026-06-24T00:00:00Z', 'usage_count': 4, 'text': 'c' * 50},
    ]
    # For 'general' role, a gets 0.9*1.2=1.08, c gets 0.7*1.2=0.84, b gets 0.5*1.0=0.5
    selected = mgr.select_by_relevance(skills, 'general', token_budget=1000)
    assert len(selected) == 3
    assert selected[0]['skill_id'] == 'a'
    assert selected[1]['skill_id'] == 'c'
    assert selected[2]['skill_id'] == 'b'
    print("  PASS: relevance selection orders by utility_score * role_bonus")


def test_token_budget_respected(tmp_path):
    skills_file = tmp_path / "skills.md"
    mgr = SkillManager(str(skills_file))
    skills = [
        {'skill_id': 'a', 'utility_score': 0.9, 'role': 'general', 'last_used': '2026-06-24T00:00:00Z', 'usage_count': 5, 'text': 'A' * 400},
        {'skill_id': 'b', 'utility_score': 0.8, 'role': 'general', 'last_used': '2026-06-24T00:00:00Z', 'usage_count': 4, 'text': 'B' * 400},
    ]
    # 400 chars ~ 105 tokens, budget 150 should only fit 1
    selected = mgr.select_by_relevance(skills, 'general', token_budget=150)
    assert len(selected) == 1
    assert selected[0]['skill_id'] == 'a'
    print("  PASS: token budget limits selection")


def test_empty_file(tmp_path):
    skills_file = tmp_path / "skills.md"
    skills_file.write_text("")
    mgr = SkillManager(str(skills_file))
    assert mgr.load() == []
    print("  PASS: empty file returns empty list")


def test_nonexistent_file(tmp_path):
    mgr = SkillManager(str(tmp_path / "nonexistent.md"))
    assert mgr.load() == []
    print("  PASS: nonexistent file returns empty list")


if __name__ == "__main__":
    import tempfile
    print("SkillManager Unit Tests:\n")
    with tempfile.TemporaryDirectory() as tmp:
        test_load_skills_with_metadata(type('tmp', (), {'path': tmp})())
    # Use system temp for tests that need tmp_path
    import atexit, shutil
    td = os.path.join(tempfile.gettempdir(), f"skilltest_{os.getpid()}")
    os.makedirs(td, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(td, ignore_errors=True))
    
    class Tmp:
        def __init__(self, name):
            self.name = name
        def __truediv__(self, other):
            return os.path.join(self.name, str(other))
    
    test_load_skills_with_metadata(Tmp(td))
    test_load_plain_text_fallback(Tmp(td))
    test_save_and_reload(Tmp(td))
    test_bayesian_update(Tmp(td))
    test_forgetting_curve(Tmp(td))
    test_pruning(Tmp(td))
    test_selection_by_relevance(Tmp(td))
    test_token_budget_respected(Tmp(td))
    test_empty_file(Tmp(td))
    test_nonexistent_file(Tmp(td))
    print("\nAll SkillManager tests passed!")
