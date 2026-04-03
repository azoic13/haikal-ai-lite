def search_books(query: str) -> tuple[str, list]:
    """
    Deep hybrid retrieval that searches ONLY inside your Chroma library.

    Stages:
    1) wide recall (semantic + keyword + exact phrase)
    2) score + rerank
    3) secondary expansion from top passages
    """
    context, sources = "", []

    try:
        if collection.count() == 0:
            st.warning("⚠️ Book database is empty.")
            return "", []

        # -----------------------------
        # PREP QUERY
        # -----------------------------
        variants = build_query_variants(query)

        translit = transliterate_to_arabic(query) if is_transliteration(query) else ""
        best_arabic = normalize_arabic(translit or query)

        stripped = re.sub(
            r"^(ما هو|ما هي|ما صحة|ما حكم|ما|هل|كيف|what is|ruling on|tell me about)\s+",
            "",
            best_arabic,
            flags=re.IGNORECASE,
        ).strip()

        keywords = get_arabic_keywords(stripped or best_arabic)

        # -----------------------------
        # 1) WIDE RECALL
        # -----------------------------
        kw_hits = keyword_search(
            keywords + ([stripped] if stripped else []),
            max_results=40
        )

        sem_hits = semantic_search(
            variants,
            n_per_variant=40
        )

        phrase_hits = []
        if stripped:
            phrase_hits = keyword_search([stripped], max_results=30)

        # -----------------------------
        # 2) MERGE + SCORE
        # -----------------------------
        all_hits = phrase_hits + kw_hits + sem_hits
        scored = []
        seen = set()

        for doc, source in all_hits:
            key = doc[:150]
            if key in seen:
                continue
            seen.add(key)

            score = 0

            # exact keyword boost
            for kw in keywords[:8]:
                if kw in doc:
                    score += 3

            # exact phrase boost
            if stripped and stripped in doc:
                score += 10

            # repeated source boost
            score += sum(1 for _, s in all_hits if s == source)

            scored.append((score, doc, source))

        scored.sort(reverse=True, key=lambda x: x[0])

        # -----------------------------
        # 3) SECONDARY EXPANSION
        # -----------------------------
        top_hits = scored[:40]
        expanded_docs = []
        seen_docs = set()

        for score, doc, source in top_hits:
            key = doc[:150]
            if key in seen_docs:
                continue

            seen_docs.add(key)
            expanded_docs.append((doc, source))

            # use longest sentence from hit to expand context
            sentences = [
                s.strip()
                for s in re.split(r"[.!؟\n]", doc)
                if s.strip()
            ]
            best_sentence = max(sentences, key=len) if sentences else ""

            if len(best_sentence) > 20:
                try:
                    extra = collection.query(
                        query_texts=[best_sentence[:300]],
                        n_results=5
                    )

                    if extra.get("documents") and extra["documents"][0]:
                        for ex_doc, ex_meta in zip(
                            extra["documents"][0],
                            extra["metadatas"][0]
                        ):
                            ex_key = ex_doc[:150]
                            if ex_key not in seen_docs:
                                seen_docs.add(ex_key)
                                expanded_docs.append(
                                    (
                                        ex_doc,
                                        ex_meta.get("source", "Unknown Book")
                                    )
                                )
                except Exception:
                    pass

            if len(expanded_docs) >= 60:
                break

        # -----------------------------
        # BUILD FINAL CONTEXT
        # -----------------------------
        for doc, source in expanded_docs[:60]:
            sources.append(source)
            context += f"\n[BOOK SOURCE: {source}]\n{doc}\n"

        if context:
            st.caption(
                f"📚 Deep search found {len(expanded_docs)} passages "
                f"from {len(set(sources))} source(s)"
            )
        else:
            st.info("ℹ️ No matching passages found in the book database.")

    except Exception as e:
        st.warning(f"⚠️ Book search error: {e}")

    return context, sources
