# BioPortal REST API Documentation Summary

## Overview
BioPortal provides a RESTful API for accessing biomedical ontologies. The API uses **hypermedia links** to navigate between resources, similar to web pages. All responses are in JSON-LD format by default.

**Base URL:** `http://data.bioontology.org`

## Authentication
- **Method:** API Key via HTTP Header
- **Header Format:** `Authorization: apikey token={YOUR_API_KEY}`
- **Your Implementation:** ✅ Correctly uses `opener.addheaders = [('Authorization', 'apikey token=' + API_KEY)]`

## Key API Concepts

### 1. Hypermedia Links
Every resource contains a `links` object with URLs to related resources:
- `self` - The resource itself
- `children` - Child classes/nodes
- `parents` - Parent classes
- `ancestors` - All ancestor classes
- `descendants` - All descendant classes
- `tree` - Full subtree
- `ontology` - The ontology this resource belongs to

**Your Implementation:** ✅ Correctly uses `node_data['links']['children']` to navigate the tree

### 2. Media Types (Resource Types)
The API supports various resource types:
- **Class** (`http://www.w3.org/2002/07/owl#Class`) - Ontology classes/concepts
- **Ontology** (`http://data.bioontology.org/metadata/Ontology`) - Entire ontologies
- **Collection** - Collections of resources
- And many others...

**Your Implementation:** ✅ Working with Classes from DOID ontology

## Main Endpoints Used in Your Notebook

### 1. Term Search
**Endpoint:** `GET /search?q={query}`

**Parameters:**
- `q` - Search query (required)
- `ontologies={id1,id2,id3}` - Filter by specific ontologies
- `require_exact_match={true|false}` - Default: false
- `suggest={true|false}` - Type-ahead suggestions
- `include={prefLabel,synonym,definition,notation,cui,semanticType}` - Fields to include
- `page={integer}` - Page number (default: 1)
- `page_size={integer}` - Results per page (default: 50)

**Your Usage:** ✅ Cell 2 uses `/search?q=heart` (basic search)

**Potential Improvements:**
- Could filter by `ontologies=DOID` to limit to Disease Ontology
- Could use `require_exact_match=true` for precise matches
- Could specify `include` to reduce response size

### 2. Ontology Access
**Endpoint:** `GET /ontologies/{acronym}`

**Example:** `/ontologies/DOID`

**Returns:**
- Ontology metadata
- Links to related resources (classes, roots, properties, etc.)

**Your Usage:** ✅ Cell 4 correctly accesses `/ontologies/DOID`

**Related Links:**
- `/ontologies/{acronym}/classes/roots` - Get root classes
- `/ontologies/{acronym}/classes` - Get all classes
- `/ontologies/{acronym}/classes/{class_id}` - Get specific class

### 3. Class Access
**Endpoint:** `GET /ontologies/{acronym}/classes/{class_id}`

**Class ID Format:** URI-encoded class identifier
- Example: `http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2FSYMP_0000462`

**Returns:**
- `prefLabel` - Preferred label (name)
- `synonym` - Array of synonyms
- `definition` - Array of definitions
- `links` - Hypermedia links to related resources

**Your Usage:** ✅ Cell 10 uses hardcoded root URL for symptom class

**Hypermedia Links Available:**
- `links.children` - Direct children
- `links.parents` - Direct parents
- `links.ancestors` - All ancestors
- `links.descendants` - All descendants
- `links.tree` - Full subtree (might be more efficient than recursive traversal!)

### 4. Children Collection
**Endpoint:** `GET /ontologies/{acronym}/classes/{class_id}/children`

**Returns:**
- `collection` - Array of child class objects
- Each child has same structure as parent (prefLabel, synonym, definition, links)

**Your Usage:** ✅ Cell 5-9 and Cell 10 use `links.children` to get children

## Advanced Features (Not Currently Used)

### 1. Subtree Search
**Endpoint:** `/search?q={query}&ontology={id}&subtree_root_id={encoded_uri}`

**Use Case:** Search only within a specific branch of the ontology
- Could search for symptoms within the symptom subtree only

### 2. Roots-Only Search
**Endpoint:** `/search?q={query}&ontologies={id1,id2}&roots_only=true`

**Use Case:** Find only root-level classes

### 3. Tree Endpoint
**Endpoint:** `/ontologies/{acronym}/classes/{class_id}/tree`

**Returns:** Full subtree in a single request (potentially more efficient than recursive traversal)

**Potential Improvement:** Could use this instead of recursive `children` calls

### 4. Batch Endpoint
**Endpoint:** `POST /batch`

**Use Case:** Retrieve multiple resources in a single request
- Reduces HTTP overhead
- Could speed up tree traversal significantly

**Example Request Body:**
```json
{
  "http://data.bioontology.org/ontologies/DOID/classes/...": {},
  "http://data.bioontology.org/ontologies/DOID/classes/...": {}
}
```

### 5. Annotator
**Endpoint:** `GET /annotator?text={input_text}`

**Use Case:** Extract ontology terms from free text
- Could be useful for validating your NER model
- Returns matched classes with positions in text

### 6. Recommender
**Endpoint:** `GET /recommender?input={text_or_keywords}`

**Use Case:** Suggest appropriate ontologies for given text/keywords

## Common Parameters

### Display/Include Parameters
- `include={field1,field2,...}` - Specify which fields to return
- `display={field1,field2,...}` - Alternative to include
- Default includes: `prefLabel`, `synonym`, `definition`, `notation`, `cui`, `semanticType`

**Potential Optimization:** Specify only needed fields to reduce response size

### Pagination
- `page={integer}` - Page number (default: 1)
- `page_size={integer}` - Results per page (default: 50, max: 100)

**Note:** Your recursive traversal doesn't handle pagination - if a class has >50 children, you'll miss some!

## Content Types

### Default: JSON-LD
- Standard JSON format
- Includes `@context` for Linked Data
- Can be parsed as normal JSON

### Alternative: XML
- Use `?format=xml` or `Accept: application/xml` header

## HTTP Verbs

- **GET** - Retrieve resources (what you're using)
- **POST** - Create resources (with server-assigned ID)
- **PUT** - Create resources (with client-assigned ID)
- **PATCH** - Modify existing resources
- **DELETE** - Delete resources

## Your Notebook Analysis

### ✅ What You're Doing Well:

1. **Authentication:** Correctly using API key in headers
2. **Hypermedia Navigation:** Properly following `links.children` to traverse tree
3. **Error Handling:** Try-except blocks in traversal function
4. **Data Extraction:** Capturing `prefLabel`, `synonym`, `definition`
5. **Export:** Saving to both JSON and CSV formats

### 🔧 Potential Improvements:

1. **Pagination:** Children collections might be paginated - you should handle this:
   ```python
   def get_all_children(children_url):
       all_children = []
       page = 1
       while True:
           page_url = f"{children_url}?page={page}&page_size=100"
           data = get_json(page_url)
           all_children.extend(data['collection'])
           if len(data['collection']) < 100:  # Last page
               break
           page += 1
       return all_children
   ```

2. **Tree Endpoint:** Consider using `/tree` endpoint instead of recursive children calls:
   ```python
   tree_url = f"{class_url}/tree"
   tree_data = get_json(tree_url)
   # Gets entire subtree in one request
   ```

3. **Batch Requests:** For large traversals, use batch endpoint:
   ```python
   # Collect all URLs first
   urls = [child['links']['self'] for child in children]
   # Then batch request
   batch_data = {"urls": urls}
   ```

4. **Rate Limiting:** Add delays between requests to avoid hitting rate limits:
   ```python
   import time
   time.sleep(0.1)  # 100ms delay between requests
   ```

5. **Caching:** Cache responses to avoid redundant API calls:
   ```python
   from functools import lru_cache
   @lru_cache(maxsize=1000)
   def get_json_cached(url):
       return get_json(url)
   ```

6. **Include Parameter:** Specify only needed fields:
   ```python
   url = f"{base_url}?include=prefLabel,synonym,definition"
   ```

7. **Error Recovery:** Add retry logic for network errors:
   ```python
   import time
   def get_json_with_retry(url, max_retries=3):
       for attempt in range(max_retries):
           try:
               return get_json(url)
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               time.sleep(2 ** attempt)  # Exponential backoff
   ```

## Key Takeaways

1. **Hypermedia Links:** The API is designed for navigation via links, not hardcoded URLs
2. **Collections:** Many endpoints return `collection` arrays that may be paginated
3. **Efficiency:** Consider using `/tree` or batch endpoints for large operations
4. **Filtering:** Use search parameters to narrow results before processing
5. **Rate Limits:** Be mindful of API rate limits - add delays if needed

## Documentation References

- **Main Documentation:** https://data.bioontology.org/documentation
- **Sample Code:** https://github.com/ncbo/ncbo_rest_sample_code
- **Support:** support@bioontology.org

## Your Specific Use Case

For extracting the symptom hierarchy from DOID:

1. ✅ **Current Approach:** Recursive traversal using children links
2. 🔧 **Optimization:** Consider using `/tree` endpoint for each major branch
3. 🔧 **Pagination:** Ensure you're getting all children (not just first 50)
4. 🔧 **Caching:** Cache responses to avoid redundant calls during development
5. 🔧 **Error Handling:** Add retry logic for network issues

Your implementation is solid and follows the API patterns correctly! The main improvements would be around efficiency (pagination, batching, caching) and robustness (error handling, rate limiting).

