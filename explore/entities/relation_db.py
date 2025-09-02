from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from typing import List, Optional, Dict, Any
import os

Base = declarative_base()


class RelationModel(Base):
    """SQLAlchemy model for Relation table"""
    __tablename__ = 'relations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    source_entity_id = Column(Integer, nullable=True, index=True)
    target_entity_id = Column(Integer, nullable=True, index=True)
    properties = Column(JSON, default={})
    confidence = Column(Float, default=1.0)
    
    # Create composite index for efficient querying
    __table_args__ = (
        Index('idx_source_target', 'source_entity_id', 'target_entity_id'),
        Index('idx_name_confidence', 'name', 'confidence'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'properties': self.properties or {},
            'confidence': self.confidence
        }
    
    def __repr__(self):
        return f"<Relation(id={self.id}, name={self.name}, source={self.source_entity_id}, target={self.target_entity_id})>"


class RelationDAO:
    """Data Access Object for Relation CRUD operations"""
    
    def __init__(self, db_url: str = None):
        """
        Initialize DAO with database connection
        
        Args:
            db_url: Database URL (e.g., 'sqlite:///relations.db' or 'mysql://user:pass@localhost/dbname')
        """
        if db_url is None:
            # Default to SQLite database
            db_path = os.path.join(os.path.dirname(__file__), '../../data/relations.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            db_url = f'sqlite:///{db_path}'
        
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def create(self, name: str, source_entity_id: int = None, 
               target_entity_id: int = None, properties: Dict = None, 
               confidence: float = 1.0) -> RelationModel:
        """
        Create a new relation
        
        Args:
            name: Relation name/type
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            properties: Additional properties as dictionary
            confidence: Confidence score (0-1)
        
        Returns:
            Created RelationModel instance
        """
        session = self.get_session()
        try:
            relation = RelationModel(
                name=name,
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                properties=properties or {},
                confidence=confidence
            )
            session.add(relation)
            session.commit()
            session.refresh(relation)
            return relation
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def read(self, relation_id: int) -> Optional[RelationModel]:
        """
        Read a relation by ID
        
        Args:
            relation_id: Relation ID
        
        Returns:
            RelationModel instance or None if not found
        """
        session = self.get_session()
        try:
            return session.query(RelationModel).filter_by(id=relation_id).first()
        finally:
            session.close()
    
    def read_all(self, limit: int = None, offset: int = 0) -> List[RelationModel]:
        """
        Read all relations with pagination
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
        
        Returns:
            List of RelationModel instances
        """
        session = self.get_session()
        try:
            query = session.query(RelationModel).offset(offset)
            if limit:
                query = query.limit(limit)
            return query.all()
        finally:
            session.close()
    
    def read_by_name(self, name: str) -> List[RelationModel]:
        """
        Read all relations with a specific name
        
        Args:
            name: Relation name
        
        Returns:
            List of RelationModel instances
        """
        session = self.get_session()
        try:
            return session.query(RelationModel).filter_by(name=name).all()
        finally:
            session.close()
    
    def read_by_entity(self, entity_id: int, as_source: bool = True) -> List[RelationModel]:
        """
        Read all relations connected to an entity
        
        Args:
            entity_id: Entity ID
            as_source: If True, find relations where entity is source; else where entity is target
        
        Returns:
            List of RelationModel instances
        """
        session = self.get_session()
        try:
            if as_source:
                return session.query(RelationModel).filter_by(source_entity_id=entity_id).all()
            else:
                return session.query(RelationModel).filter_by(target_entity_id=entity_id).all()
        finally:
            session.close()
    
    def find_relations_between(self, source_id: int, target_id: int) -> List[RelationModel]:
        """
        Find all relations between two entities
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
        
        Returns:
            List of RelationModel instances
        """
        session = self.get_session()
        try:
            return session.query(RelationModel).filter_by(
                source_entity_id=source_id,
                target_entity_id=target_id
            ).all()
        finally:
            session.close()
    
    def update(self, relation_id: int, **kwargs) -> Optional[RelationModel]:
        """
        Update a relation
        
        Args:
            relation_id: Relation ID
            **kwargs: Fields to update (name, source_entity_id, target_entity_id, properties, confidence)
        
        Returns:
            Updated RelationModel instance or None if not found
        """
        session = self.get_session()
        try:
            relation = session.query(RelationModel).filter_by(id=relation_id).first()
            if not relation:
                return None
            
            for key, value in kwargs.items():
                if hasattr(relation, key):
                    setattr(relation, key, value)
            
            session.commit()
            session.refresh(relation)
            return relation
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def delete(self, relation_id: int) -> bool:
        """
        Delete a relation by ID
        
        Args:
            relation_id: Relation ID
        
        Returns:
            True if deleted, False if not found
        """
        session = self.get_session()
        try:
            relation = session.query(RelationModel).filter_by(id=relation_id).first()
            if not relation:
                return False
            
            session.delete(relation)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def delete_by_entity(self, entity_id: int) -> int:
        """
        Delete all relations connected to an entity
        
        Args:
            entity_id: Entity ID
        
        Returns:
            Number of relations deleted
        """
        session = self.get_session()
        try:
            deleted_count = session.query(RelationModel).filter(
                (RelationModel.source_entity_id == entity_id) |
                (RelationModel.target_entity_id == entity_id)
            ).delete()
            session.commit()
            return deleted_count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def bulk_create(self, relations: List[Dict[str, Any]]) -> List[RelationModel]:
        """
        Create multiple relations in a single transaction
        
        Args:
            relations: List of relation dictionaries
        
        Returns:
            List of created RelationModel instances
        """
        session = self.get_session()
        try:
            relation_models = []
            for rel_data in relations:
                relation = RelationModel(
                    name=rel_data.get('name'),
                    source_entity_id=rel_data.get('source_entity_id'),
                    target_entity_id=rel_data.get('target_entity_id'),
                    properties=rel_data.get('properties', {}),
                    confidence=rel_data.get('confidence', 1.0)
                )
                session.add(relation)
                relation_models.append(relation)
            
            session.commit()
            for rel in relation_models:
                session.refresh(rel)
            return relation_models
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def search(self, name_pattern: str = None, min_confidence: float = None,
               properties_filter: Dict = None) -> List[RelationModel]:
        """
        Search relations with filters
        
        Args:
            name_pattern: Pattern to match relation name (supports SQL LIKE wildcards)
            min_confidence: Minimum confidence score
            properties_filter: Dictionary of properties to filter
        
        Returns:
            List of matching RelationModel instances
        """
        session = self.get_session()
        try:
            query = session.query(RelationModel)
            
            if name_pattern:
                query = query.filter(RelationModel.name.like(name_pattern))
            
            if min_confidence is not None:
                query = query.filter(RelationModel.confidence >= min_confidence)
            
            if properties_filter:
                for key, value in properties_filter.items():
                    # This works for PostgreSQL and MySQL with JSON support
                    # For SQLite, you might need different approach
                    query = query.filter(
                        RelationModel.properties[key].astext == str(value)
                    )
            
            return query.all()
        finally:
            session.close()
    
    def count(self, name: str = None) -> int:
        """
        Count relations
        
        Args:
            name: Optional relation name to filter by
        
        Returns:
            Number of relations
        """
        session = self.get_session()
        try:
            query = session.query(RelationModel)
            if name:
                query = query.filter_by(name=name)
            return query.count()
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about relations
        
        Returns:
            Dictionary with statistics
        """
        session = self.get_session()
        try:
            from sqlalchemy import func
            
            total_count = session.query(RelationModel).count()
            
            # Count by relation name
            name_counts = session.query(
                RelationModel.name,
                func.count(RelationModel.id).label('count')
            ).group_by(RelationModel.name).all()
            
            # Average confidence
            avg_confidence = session.query(
                func.avg(RelationModel.confidence)
            ).scalar()
            
            return {
                'total_relations': total_count,
                'relations_by_name': {name: count for name, count in name_counts},
                'average_confidence': float(avg_confidence) if avg_confidence else 0,
                'unique_relation_types': len(name_counts)
            }
        finally:
            session.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize DAO with SQLite database
    dao = RelationDAO('sqlite:///test_relations.db')
    
    # Create relations
    rel1 = dao.create(
        name="located_in",
        source_entity_id=1,
        target_entity_id=2,
        properties={"since": "2020", "type": "primary"},
        confidence=0.95
    )
    print(f"Created: {rel1}")
    
    rel2 = dao.create(
        name="works_for",
        source_entity_id=3,
        target_entity_id=4,
        properties={"position": "engineer", "department": "R&D"},
        confidence=0.88
    )
    print(f"Created: {rel2}")
    
    # Bulk create
    bulk_relations = [
        {
            'name': 'knows',
            'source_entity_id': 1,
            'target_entity_id': 3,
            'confidence': 0.75
        },
        {
            'name': 'located_in',
            'source_entity_id': 4,
            'target_entity_id': 2,
            'confidence': 0.92
        }
    ]
    bulk_created = dao.bulk_create(bulk_relations)
    print(f"Bulk created {len(bulk_created)} relations")
    
    # Read operations
    found_rel = dao.read(rel1.id)
    print(f"Found by ID: {found_rel}")
    
    same_name_rels = dao.read_by_name("located_in")
    print(f"Found {len(same_name_rels)} relations named 'located_in'")
    
    entity_rels = dao.read_by_entity(1, as_source=True)
    print(f"Entity 1 has {len(entity_rels)} outgoing relations")
    
    # Update operation
    updated_rel = dao.update(
        rel1.id,
        confidence=0.98,
        properties={"since": "2020", "type": "primary", "verified": True}
    )
    print(f"Updated: {updated_rel}")
    
    # Search operations
    high_confidence_rels = dao.search(min_confidence=0.9)
    print(f"Found {len(high_confidence_rels)} high confidence relations")
    
    # Get statistics
    stats = dao.get_statistics()
    print(f"Statistics: {stats}")
    
    # Find relations between entities
    between_rels = dao.find_relations_between(1, 2)
    print(f"Relations between entity 1 and 2: {len(between_rels)}")
    
    # Count operations
    total_count = dao.count()
    located_count = dao.count(name="located_in")
    print(f"Total: {total_count}, Located_in: {located_count}")
    
    # Delete operation
    deleted = dao.delete(rel2.id)
    print(f"Deleted relation: {deleted}")
    
    # Clean up test database
    import os
    if os.path.exists('test_relations.db'):
        os.remove('test_relations.db')
        print("Test database cleaned up")