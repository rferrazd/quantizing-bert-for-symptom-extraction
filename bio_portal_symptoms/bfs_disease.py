#!/usr/bin/env python3
"""
BFS Traversal of DOID (Human Disease Ontology) with all improvements:
- Timeout handling (prevents hanging)
- Retry logic with exponential backoff
- Resume capability (loads existing CSV)
- Frequent checkpointing (saves every 10 nodes)
- Detailed logging (shows progress and where it's stuck)
- Proper pagination handling
"""

import urllib.request, urllib.error, urllib.parse
import json
import os
import time
import pandas as pd
from collections import deque
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURATION ---
REST_URL = "http://data.bioontology.org"
OUTPUT_FILE = "doid_disease_tree.csv"
SAVE_INTERVAL = 10  # Save every 10 nodes (frequent checkpoints)
API_DELAY = 0.05  # Rate limiting between requests
API_TIMEOUT = 30  # Timeout for API calls (prevents hanging)
MAX_RETRIES = 3  # Retry failed requests up to 3 times

# Load API key from .env file
load_dotenv()
API_KEY = os.environ.get("BIO_PORTAL_API_KEY", "")

if not API_KEY:
    raise ValueError("BIO_PORTAL_API_KEY not found in environment. Please set it in .env file")


def get_json(url, timeout=API_TIMEOUT):
    """
    Fetch JSON with timeout to prevent hanging.
    This is the CRITICAL fix - without timeout, API calls can hang forever.
    """
    opener = urllib.request.build_opener()
    opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]
    try:
        response = opener.open(url, timeout=timeout)
        return json.loads(response.read())
    except urllib.error.URLError as e:
        print(f"⚠️ URL Error for {url}: {e}")
        raise
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        raise


def get_json_with_retry(url, retries=MAX_RETRIES, timeout=API_TIMEOUT):
    """
    Enhanced get_json with timeout and retry logic.
    Prevents hanging and handles transient failures.
    """
    for attempt in range(retries):
        try:
            time.sleep(API_DELAY)  # Rate limiting
            return get_json(url, timeout=timeout)
        except urllib.error.URLError as e:
            if attempt == retries - 1:
                print(f"❌ Failed to fetch {url} after {retries} attempts: {e}")
                return None
            wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
            print(f"⚠️ Error fetching {url}, retrying in {wait_time}s ({attempt+1}/{retries})...")
            time.sleep(wait_time)
        except Exception as e:
            if attempt == retries - 1:
                print(f"❌ Unexpected error fetching {url}: {e}")
                return None
            wait_time = (attempt + 1) * 2
            print(f"⚠️ Error fetching {url}, retrying in {wait_time}s ({attempt+1}/{retries})...")
            time.sleep(wait_time)
    return None


def get_all_children_pagination(children_url):
    """
    Get ALL children using proper BioPortal pagination with timeout/retry.
    Uses the 'nextPage' link pattern from official sample code.
    Includes detailed logging to see where pagination might get stuck.
    FIX: Stops pagination when encountering empty pages to prevent infinite loops.
    """
    all_children = []
    MAX_PAGES = 100  # Safety limit to prevent infinite loops
    consecutive_empty_pages = 0
    
    # Get first page with retry/timeout
    print(f"   ↳ [PAGINATION] Starting: page 1...")
    page = get_json_with_retry(children_url)
    if not page:
        print(f"   ⚠️ [PAGINATION] Failed to fetch first page")
        return []
    
    # Process first page
    if 'collection' in page:
        items_in_page = len(page['collection'])
        all_children.extend(page['collection'])
        print(f"   ↳ [PAGINATION] Page 1: {items_in_page} items (Total: {len(all_children)})")
        
        # If first page is empty, stop immediately
        if items_in_page == 0:
            print(f"   ✅ [PAGINATION] First page empty, no children")
            return []
    else:
        print(f"   ✅ [PAGINATION] No collection in first page, no children")
        return []
    
    # Follow nextPage links
    next_page_url = page.get("links", {}).get("nextPage")
    page_count = 1
    
    while next_page_url and page_count < MAX_PAGES:
        page_count += 1
        print(f"   ↳ [PAGINATION] Fetching page {page_count}...")
        page = get_json_with_retry(next_page_url)
        if not page:
            print(f"   ⚠️ [PAGINATION] Failed to fetch page {page_count}, stopping pagination")
            break
        
        items_in_page = 0
        if 'collection' in page:
            items_in_page = len(page['collection'])
            all_children.extend(page['collection'])
            print(f"   ↳ [PAGINATION] Page {page_count}: {items_in_page} items (Total: {len(all_children)})")
        
        # Stop if we get an empty page (API bug: returns nextPage even when no items)
        if items_in_page == 0:
            print(f"   ⚠️ [PAGINATION] Page {page_count} is empty, stopping (API may return empty pages with nextPage)")
            break
        
        next_page_url = page.get("links", {}).get("nextPage")
    
    if page_count >= MAX_PAGES:
        print(f"   ⚠️ [PAGINATION] Reached max pages limit ({MAX_PAGES}), stopping")
    
    print(f"   ✅ [PAGINATION] Complete: {len(all_children)} total children from {page_count} pages")
    return all_children


def load_existing_checkpoint(filename):
    """
    Load existing CSV to resume from checkpoint.
    Returns: (visited_ids set, existing_data list, processed_count)
    """
    visited_ids = set()
    existing_data = []
    
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            visited_ids = set(df['id'].dropna().astype(str).tolist())
            existing_data = df.to_dict('records')
            print(f"📂 Loaded checkpoint: {len(existing_data)} existing records")
            print(f"   {len(visited_ids)} unique node IDs will be skipped")
            return visited_ids, existing_data, len(existing_data)
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
            print("   Starting fresh...")
    else:
        print(f"📝 No existing checkpoint. Starting fresh.")
    
    return visited_ids, existing_data, 0


def build_disease_tree(root_node, output_file, resume=True):
    """
    IMPROVED BFS traversal with all best practices:
    - Retry logic with exponential backoff
    - Timeout handling (prevents hanging)
    - Resume capability (loads existing CSV)
    - Frequent checkpoints (every 10 nodes)
    - Detailed progress logging
    - Proper error handling
    """
    
    # Load checkpoint if resuming
    if resume:
        visited_ids, existing_data, processed_count = load_existing_checkpoint(output_file)
        database = existing_data.copy()
        print(f"🔄 Resuming from checkpoint: {processed_count} nodes already processed")
    else:
        visited_ids = set()
        database = []
        processed_count = 0
        print("🆕 Starting fresh traversal")
    
    # Initialize queue
    queue = deque()
    
    # Get full root node info
    if 'links' in root_node and 'self' in root_node['links']:
        root_full = get_json_with_retry(root_node['links']['self'])
        if not root_full:
            print("❌ Could not fetch root node. Exiting.")
            return pd.DataFrame(database)
    else:
        root_full = root_node
    
    root_id = root_full.get('@id', '')
    root_label = root_full.get('prefLabel', 'disease')
    root_path = root_label
    
    # Add root to queue if not already processed
    if root_id and root_id not in visited_ids:
        queue.append((root_full, root_path, 0))
        visited_ids.add(root_id)
        print(f"✅ Root node '{root_label}' added to queue")
    else:
        print(f"⏭️ Root node '{root_label}' already processed, checking children...")
        # If root processed, we need to rebuild queue from its unprocessed children
        if 'links' in root_full and 'children' in root_full['links']:
            children = get_all_children_pagination(root_full['links']['children'])
            for child in children:
                child_id = child.get('@id')
                if child_id and child_id not in visited_ids:
                    child_label = child.get('prefLabel', 'Unknown')
                    queue.append((child, f"{root_path}/{child_label}", 1))
                    visited_ids.add(child_id)
    
    print(f"🚀 Starting BFS traversal")
    print(f"📂 Output file: {output_file}")
    print(f"💾 Checkpoint interval: Every {SAVE_INTERVAL} nodes")
    print(f"📊 Queue size: {len(queue)}")
    print("-" * 70)
    
    start_time = time.time()
    last_save_time = time.time()
    
    try:
        while queue:
            current_node, current_path, current_level = queue.popleft()
            
            node_id = current_node.get('@id', '')
            node_label = current_node.get('prefLabel', 'Unknown')
            
            # Skip if already processed (safety check)
            if node_id in visited_ids and any(r.get('id') == node_id for r in database):
                continue
            
            # Create record
            record = {
                'id': node_id,
                'level': current_level,
                'path': current_path,
                'prefLabel': node_label
            }
            database.append(record)
            visited_ids.add(node_id)
            processed_count += 1
            
            # Progress logging (every 5 nodes or first 50)
            if processed_count % 5 == 0 or processed_count <= 50:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"[{processed_count:5d}] L{current_level} | Q:{len(queue):4d} | "
                      f"{node_label[:45]:45s} | {rate:.2f} nodes/sec")
            
            # Frequent checkpoints
            if processed_count % SAVE_INTERVAL == 0:
                df_checkpoint = pd.DataFrame(database)
                df_checkpoint.to_csv(output_file, index=False)
                elapsed = time.time() - start_time
                save_time = time.time() - last_save_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"💾 CHECKPOINT: Saved {processed_count} nodes | "
                      f"Rate: {rate:.2f} nodes/sec | Elapsed: {elapsed:.0f}s | "
                      f"Save took: {save_time:.1f}s")
                last_save_time = time.time()
            
            # Fetch children
            if 'links' in current_node and 'children' in current_node['links']:
                children_url = current_node['links']['children']
                
                # LOG EVERY NODE (not just every 10th) - helps diagnose where it's stuck
                print(f"[{processed_count:5d}] ↳ Fetching children for: {node_label[:50]} (L{current_level})")
                
                try:
                    children = get_all_children_pagination(children_url)
                    
                    if not children:
                        print(f"[{processed_count:5d}] ⚠️ No children or fetch failed")
                    else:
                        print(f"[{processed_count:5d}] ✅ Found {len(children)} children")
                    
                    for child in children:
                        child_id = child.get('@id')
                        child_label = child.get('prefLabel', 'Unknown')
                        
                        if child_id and child_id not in visited_ids:
                            visited_ids.add(child_id)
                            new_path = f"{current_path}/{child_label}"
                            queue.append((child, new_path, current_level + 1))
                            
                except Exception as e:
                    print(f"❌ Error fetching children for '{node_label}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    except KeyboardInterrupt:
        print("\n\n🛑 Traversal interrupted by user!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final save
        print("\n📝 Saving final data...")
        df_final = pd.DataFrame(database)
        df_final.to_csv(output_file, index=False)
        total_time = time.time() - start_time
        rate = processed_count / total_time if total_time > 0 else 0
        print(f"✅ COMPLETE: Saved {len(df_final)} nodes to {output_file}")
        print(f"⏱️  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"📊 Average rate: {rate:.2f} nodes/sec")
        return df_final


def get_disease_root():
    """
    Get the disease root node from DOID ontology.
    """
    print("=" * 50)
    print("FETCHING DISEASE ROOT FROM DOID ONTOLOGY")
    print("=" * 50)
    
    # Access the DOID ontology
    doid_acronym = "DOID"
    doid_url = f"{REST_URL}/ontologies/{doid_acronym}"
    doid_ontology = get_json_with_retry(doid_url)
    
    if not doid_ontology:
        raise RuntimeError("Could not fetch DOID ontology")
    
    print(f"✅ Accessing DOID: {doid_ontology['name']}")
    
    # Get root classes
    roots_url = doid_ontology['links']['roots']
    roots = get_json_with_retry(roots_url)
    
    if not roots:
        raise RuntimeError("Could not fetch root classes")
    
    # Find disease root
    disease_root = None
    if isinstance(roots, list):
        for root in roots:
            if root.get('prefLabel') == 'disease':
                disease_root = root
                break
    elif isinstance(roots, dict) and roots.get('prefLabel') == 'disease':
        disease_root = roots
    
    if not disease_root:
        raise RuntimeError("Could not find 'disease' root in DOID ontology")
    
    print(f"✅ Found disease root: {disease_root.get('prefLabel')}")
    return disease_root


def main():
    """
    Main execution function.
    """
    print("=" * 50)
    print("DOID DISEASE TREE BFS TRAVERSAL")
    print("=" * 50)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Timeout: {API_TIMEOUT}s | Retries: {MAX_RETRIES} | Checkpoint: Every {SAVE_INTERVAL} nodes")
    print("=" * 50)
    print()
    
    try:
        # Get disease root
        disease_root = get_disease_root()
        print()
        
        # Run the improved traversal
        # Set resume=True to continue from existing CSV, or resume=False to start fresh
        df_result = build_disease_tree(disease_root, OUTPUT_FILE, resume=True)
        
        print("\n" + "=" * 50)
        print("PREVIEW OF RESULTS")
        print("=" * 50)
        print(df_result.head(20))
        print(f"\nTotal rows: {len(df_result)}")
        print(f"Levels: {df_result['level'].min()} to {df_result['level'].max()}")
        print(f"Unique paths: {df_result['path'].nunique()}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user. Progress saved to CSV.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

