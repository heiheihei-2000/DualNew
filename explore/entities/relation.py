import json
import pickle
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import os


@dataclass
class Relation:
    """Relation entity class for knowledge graph relations"""
    
    id: Optional[int] = None
    name: str = ""
    source_entity_id: Optional[int] = None
    target_entity_id: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'properties': self.properties,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relation':
        """Create relation from dictionary"""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"Relation(id={self.id}, name={self.name}, source={self.source_entity_id}, target={self.target_entity_id})"


class RelationRepository:
    """Repository class for Relation CRUD operations"""
    
    def __init__(self, storage_path: str = "../data/relations"):
        self.storage_path = storage_path
        self.relations: Dict[int, Relation] = {}
        self.relation_name_index: Dict[str, List[int]] = {}
        self.next_id = 1
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing relations
        self._load_relations()
    
    def _load_relations(self):
        """Load relations from storage"""
        relations_file = os.path.join(self.storage_path, "relations.pkl")
        if os.path.exists(relations_file):
            with open(relations_file, 'rb') as f:
                data = pickle.load(f)
                self.relations = data.get('relations', {})
                self.relation_name_index = data.get('relation_name_index', {})
                self.next_id = data.get('next_id', 1)
    
    def _save_relations(self):
        """Save relations to storage"""
        relations_file = os.path.join(self.storage_path, "relations.pkl")
        data = {
            'relations': self.relations,
            'relation_name_index': self.relation_name_index,
            'next_id': self.next_id
        }
        with open(relations_file, 'wb') as f:
            pickle.dump(data, f)
    
    def create(self, relation: Relation) -> Relation:
        """Create a new relation"""
        if relation.id is None:
            relation.id = self.next_id
            self.next_id += 1
        
        self.relations[relation.id] = relation
        
        # Update name index
        if relation.name not in self.relation_name_index:
            self.relation_name_index[relation.name] = []
        self.relation_name_index[relation.name].append(relation.id)
        
        self._save_relations()
        return relation
    
    def read(self, relation_id: int) -> Optional[Relation]:
        """Read a relation by ID"""
        return self.relations.get(relation_id)
    
    def read_by_name(self, name: str) -> List[Relation]:
        """Read all relations with a specific name"""
        relation_ids = self.relation_name_index.get(name, [])
        return [self.relations[rid] for rid in relation_ids if rid in self.relations]
    
    def read_by_entity(self, entity_id: int, as_source: bool = True) -> List[Relation]:
        """Read all relations connected to an entity"""
        results = []
        for relation in self.relations.values():
            if as_source and relation.source_entity_id == entity_id:
                results.append(relation)
            elif not as_source and relation.target_entity_id == entity_id:
                results.append(relation)
        return results
    
    def read_all(self) -> List[Relation]:
        """Read all relations"""
        return list(self.relations.values())
    
    def update(self, relation: Relation) -> Optional[Relation]:
        """Update an existing relation"""
        if relation.id not in self.relations:
            return None
        
        old_relation = self.relations[relation.id]
        
        # Update name index if name changed
        if old_relation.name != relation.name:
            # Remove from old index
            if old_relation.name in self.relation_name_index:
                self.relation_name_index[old_relation.name].remove(relation.id)
                if not self.relation_name_index[old_relation.name]:
                    del self.relation_name_index[old_relation.name]
            
            # Add to new index
            if relation.name not in self.relation_name_index:
                self.relation_name_index[relation.name] = []
            self.relation_name_index[relation.name].append(relation.id)
        
        self.relations[relation.id] = relation
        self._save_relations()
        return relation
    
    def delete(self, relation_id: int) -> bool:
        """Delete a relation by ID"""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        # Remove from name index
        if relation.name in self.relation_name_index:
            self.relation_name_index[relation.name].remove(relation_id)
            if not self.relation_name_index[relation.name]:
                del self.relation_name_index[relation.name]
        
        del self.relations[relation_id]
        self._save_relations()
        return True
    
    def delete_by_entity(self, entity_id: int) -> int:
        """Delete all relations connected to an entity"""
        relations_to_delete = []
        for rid, relation in self.relations.items():
            if relation.source_entity_id == entity_id or relation.target_entity_id == entity_id:
                relations_to_delete.append(rid)
        
        deleted_count = 0
        for rid in relations_to_delete:
            if self.delete(rid):
                deleted_count += 1
        
        return deleted_count
    
    def find_relations_between(self, source_id: int, target_id: int) -> List[Relation]:
        """Find all relations between two entities"""
        results = []
        for relation in self.relations.values():
            if relation.source_entity_id == source_id and relation.target_entity_id == target_id:
                results.append(relation)
        return results
    
    def count(self) -> int:
        """Count total number of relations"""
        return len(self.relations)
    
    def count_by_name(self, name: str) -> int:
        """Count relations with a specific name"""
        return len(self.relation_name_index.get(name, []))
    
    def export_to_json(self, filepath: str):
        """Export all relations to JSON file"""
        data = [relation.to_dict() for relation in self.relations.values()]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_json(self, filepath: str):
        """Import relations from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            relation = Relation.from_dict(item)
            self.create(relation)
    
    def clear_all(self):
        """Clear all relations"""
        self.relations.clear()
        self.relation_name_index.clear()
        self.next_id = 1
        self._save_relations()


# Example usage and testing
if __name__ == "__main__":
    # Initialize repository
    repo = RelationRepository()
    
    # Create relations
    rel1 = Relation(
        name="located_in",
        source_entity_id=1,
        target_entity_id=2,
        properties={"since": "2020", "type": "primary"},
        confidence=0.95
    )
    created_rel1 = repo.create(rel1)
    print(f"Created: {created_rel1}")
    
    rel2 = Relation(
        name="works_for",
        source_entity_id=3,
        target_entity_id=4,
        properties={"position": "engineer"},
        confidence=0.88
    )
    created_rel2 = repo.create(rel2)
    print(f"Created: {created_rel2}")
    
    # Read operations
    found_rel = repo.read(created_rel1.id)
    print(f"Found by ID: {found_rel}")
    
    same_name_rels = repo.read_by_name("located_in")
    print(f"Found by name 'located_in': {len(same_name_rels)} relations")
    
    entity_rels = repo.read_by_entity(1, as_source=True)
    print(f"Relations from entity 1: {len(entity_rels)} relations")
    
    # Update operation
    created_rel1.confidence = 0.98
    created_rel1.properties["verified"] = True
    updated_rel = repo.update(created_rel1)
    print(f"Updated: {updated_rel}")
    
    # Find relations between entities
    between_rels = repo.find_relations_between(1, 2)
    print(f"Relations between entity 1 and 2: {len(between_rels)} relations")
    
    # Count operations
    print(f"Total relations: {repo.count()}")
    print(f"Relations named 'located_in': {repo.count_by_name('located_in')}")
    
    # Export to JSON
    repo.export_to_json("relations_export.json")
    print("Exported to JSON")
    
    # Delete operation
    deleted = repo.delete(created_rel2.id)
    print(f"Deleted relation: {deleted}")
    
    print(f"Total relations after deletion: {repo.count()}")