import json
import os
import random
from datetime import datetime, timedelta

def run_analysis():
    """Run sample astronomy analysis and save results"""
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Generate sample astronomy data
    celestial_objects = ['Star', 'Planet', 'Galaxy', 'Nebula', 'Black Hole', 'Asteroid', 'Comet']
    constellations = ['Orion', 'Ursa Major', 'Cassiopeia', 'Cygnus', 'Draco', 'Lyra', 'Hercules']
    
    # Create sample observations
    observations = []
    for i in range(100):
        observation = {
            'id': f'OBJ-{i+1:04d}',
            'object_type': random.choice(celestial_objects),
            'constellation': random.choice(constellations),
            'brightness': round(random.uniform(-30, 15), 2),
            'distance_ly': round(random.uniform(1, 50000), 2),
            'discovery_date': (datetime.now() - timedelta(days=random.randint(0, 5000))).strftime('%Y-%m-%d'),
            'confidence': round(random.uniform(0.6, 1.0), 3),
            'mass_sun': round(random.uniform(0.1, 50), 2)
        }
        observations.append(observation)
    
    # Calculate statistics
    stats = {
        'total_observations': len(observations),
        'avg_brightness': round(sum(o['brightness'] for o in observations) / len(observations), 2),
        'avg_distance': round(sum(o['distance_ly'] for o in observations) / len(observations), 2),
        'avg_confidence': round(sum(o['confidence'] for o in observations) / len(observations), 3),
        'object_counts': {},
        'constellation_counts': {}
    }
    
    # Count objects by type
    for obj in celestial_objects:
        count = sum(1 for o in observations if o['object_type'] == obj)
        stats['object_counts'][obj] = count
    
    # Count by constellation
    for const in constellations:
        count = sum(1 for o in observations if o['constellation'] == const)
        stats['constellation_counts'][const] = count
    
    # Compile final results
    results = {
        'generated_date': datetime.now().isoformat(),
        'statistics': stats,
        'recent_discoveries': observations[:15],
        'all_observations': observations[:50]
    }
    
    # Save to JSON file
    output_path = '../results/results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Analysis complete! Results saved to: {output_path}")
    print(f"📊 Generated {len(observations)} astronomical observations")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting space astronomy analysis...")
    print("=" * 50)
    run_analysis()
    print("=" * 50)
    print("✨ Analysis completed successfully!")
