"""Agent state management"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class Step:
    """Um passo no loop ReAct do agente"""
    type: str  # "thought", "action", "observation", "suspension"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Estado completo do agente durante execução"""
    goal: str
    context: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    current_step: int = 0
    status: str = "running"  # running, awaiting_approval, completed, failed
    pending_action: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def add_step(self, step: Step) -> None:
        """Adiciona um passo ao histórico"""
        self.steps.append(step)
    
    def add_observation(self, observation: str) -> None:
        """Adiciona uma observação ao contexto"""
        self.observations.append(observation)
    
    def get_recent_observations(self, n: int = 5) -> List[str]:
        """Retorna as últimas n observações"""
        return self.observations[-n:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict para serialização"""
        return {
            "goal": self.goal,
            "context": self.context,
            "steps": [{"type": s.type, "content": s.content} for s in self.steps],
            "status": self.status,
            "current_step": self.current_step
        }
