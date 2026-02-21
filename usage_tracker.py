import json
from datetime import datetime, UTC
from sqlmodel import Session, select
from sqlalchemy import func
from typing import Dict, List
from database import engine
from models import UsageStats, User


class UsageTracker:
    @staticmethod
    def track_message(user_id: int, tokens_used: int = 0, model_name: str = None, web_search_used: bool = False):
        """Track a message sent by the user"""
        with Session(engine) as session:
            today = datetime.now(UTC).date()
            
            # Get or create today's usage stats
            statement = select(UsageStats).where(
                UsageStats.user_id == user_id,
                func.date(UsageStats.date) == today
            )
            usage_stat = session.exec(statement).first()
            
            if not usage_stat:
                usage_stat = UsageStats(
                    user_id=user_id,
                    date=datetime.now(UTC),
                    messages_sent=0,
                    tokens_used=0,
                    sessions_created=0,
                    web_searches_made=0,
                    model_usage="{}"
                )
                session.add(usage_stat)
            
            # Update stats
            usage_stat.messages_sent += 1
            usage_stat.tokens_used += tokens_used
            
            if web_search_used:
                usage_stat.web_searches_made += 1
            
            # Update model usage
            model_usage = json.loads(usage_stat.model_usage) if usage_stat.model_usage else {}
            if model_name:
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
                usage_stat.model_usage = json.dumps(model_usage)
            
            session.commit()
    
    @staticmethod
    def track_session_created(user_id: int):
        """Track a new session creation"""
        with Session(engine) as session:
            today = datetime.now(UTC).date()
            
            # Get or create today's usage stats
            statement = select(UsageStats).where(
                UsageStats.user_id == user_id,
                func.date(UsageStats.date) == today
            )
            usage_stat = session.exec(statement).first()
            
            if not usage_stat:
                usage_stat = UsageStats(
                    user_id=user_id,
                    date=datetime.now(UTC),
                    messages_sent=0,
                    tokens_used=0,
                    sessions_created=0,
                    web_searches_made=0,
                    model_usage="{}"
                )
                session.add(usage_stat)
            
            usage_stat.sessions_created += 1
            session.commit()
    
    @staticmethod
    def get_user_usage_stats(user_id: int, days: int = 30) -> List[Dict]:
        """Get usage statistics for a user over the last N days"""
        with Session(engine) as session:
            # Calculate the date N days ago
            from datetime import timedelta
            start_date = datetime.now(UTC).date() - timedelta(days=days)
            
            statement = select(UsageStats).where(
                UsageStats.user_id == user_id,
                func.date(UsageStats.date) >= start_date
            ).order_by(UsageStats.date.desc())
            
            usage_stats = session.exec(statement).all()
            
            result = []
            for stat in usage_stats:
                model_usage = json.loads(stat.model_usage) if stat.model_usage else {}
                result.append({
                    "date": stat.date.strftime("%Y-%m-%d"),
                    "messages_sent": stat.messages_sent,
                    "tokens_used": stat.tokens_used,
                    "sessions_created": stat.sessions_created,
                    "web_searches_made": stat.web_searches_made,
                    "model_usage": model_usage
                })
            
            return result
    
    @staticmethod
    def get_user_total_stats(user_id: int) -> Dict:
        """Get total usage statistics for a user"""
        with Session(engine) as session:
            statement = select(
                func.sum(UsageStats.messages_sent).label("total_messages"),
                func.sum(UsageStats.tokens_used).label("total_tokens"),
                func.sum(UsageStats.sessions_created).label("total_sessions"),
                func.sum(UsageStats.web_searches_made).label("total_searches")
            ).where(UsageStats.user_id == user_id)
            
            result = session.exec(statement).first()
            
            # Get aggregated model usage
            model_statement = select(UsageStats.model_usage).where(UsageStats.user_id == user_id)
            model_usages = session.exec(model_statement).all()
            
            aggregated_models = {}
            for usage_json in model_usages:
                if usage_json:
                    model_data = json.loads(usage_json)
                    for model, count in model_data.items():
                        aggregated_models[model] = aggregated_models.get(model, 0) + count
            
            return {
                "total_messages": result[0] or 0,
                "total_tokens": result[1] or 0,
                "total_sessions": result[2] or 0,
                "total_searches": result[3] or 0,
                "model_usage": aggregated_models
            }
