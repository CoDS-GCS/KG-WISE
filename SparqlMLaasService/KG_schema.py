import argparse
import os
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd


def ensure_output_dir(output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Extract SPARQL schema statistics")
    parser.add_argument(
        "--graph-uri",
        default="http://dblp.org",
        help="Named graph URI to query (default: http://dblp.org)"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:1234/sparql",
        help="SPARQL endpoint URL (default: http://localhost:1234/sparql)"
    )
    parser.add_argument(
        "--output",
        default="logs/sparql_schema.tsv",
        help="Output TSV file location (default: logs/sparql_schema.tsv)"
    )

    args = parser.parse_args()

    ensure_output_dir(args.output)

    sparql = SPARQLWrapper(args.endpoint)
    data = []

    # ---- Query 1: Object has RDF type ----
    sparql.setQuery(f"""
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?subject_type ?p ?object_type (COUNT(*) AS ?count)
    FROM <{args.graph_uri}>
    WHERE {{
        ?s ?p ?o .
        ?s a ?subject_type .
        ?o a ?object_type .
    }}
    GROUP BY ?subject_type ?p ?object_type
    ORDER BY DESC(?count)
    """)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        data.append((
            result["subject_type"]["value"],
            result["p"]["value"],
            result["object_type"]["value"],
            result["count"]["value"]
        ))

    # ---- Query 2: Object has NO RDF type (treated as string) ----
    sparql.setQuery(f"""
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?subject_type ?p "str" AS ?object_type (COUNT(*) AS ?count)
    FROM <{args.graph_uri}>
    WHERE {{
        ?s ?p ?o .
        ?s a ?subject_type .
        FILTER NOT EXISTS {{ ?o a ?some_type }}
    }}
    GROUP BY ?subject_type ?p ?object_type
    ORDER BY DESC(?count)
    """)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        data.append((
            result["subject_type"]["value"],
            result["p"]["value"],
            result["object_type"]["value"],
            result["count"]["value"]
        ))

    # ---- Save output ----
    df = pd.DataFrame(data, columns=["s", "p", "o", "count"])
    df = df.sort_values(by="count", ascending=False)

    df.to_csv(args.output, sep="\t", header=False, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
