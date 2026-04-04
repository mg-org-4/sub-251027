# Plan de refacto complet — ComfyUI-Majoor-AssetsManager

> Dernière mise à jour : 2026-04-03
> Statut global : migration frontend Vue largement réalisée, plan maître réadapté au code actuel

## 1. But du document

Ce document devient le **plan maître de refacto** du projet.

Il ne repart pas de zéro. Il tient compte de l’état réel du repo :

- la migration majeure du frontend vers Vue 3 + Pinia est déjà en grande partie faite ;
- le runtime panel a déjà commencé à être redécoupé ;
- plusieurs modules backend sont déjà mieux structurés qu’au début du chantier ;
- il reste néanmoins des zones trop denses côté backend, viewer runtime, routes et gouvernance technique.

Ce plan sert à :

- garder une vision claire de ce qui est déjà terminé ;
- éviter de continuer à suivre des objectifs frontend devenus obsolètes ;
- prioriser les vrais points de friction restants ;
- ajouter un **suivi vérifiable par cases à cocher** ;
- faciliter les revues de progression sans devoir reconstituer l’historique à chaque fois.

---

## 2. Comment utiliser ce plan

### 2.1 Règle de lecture

- La section `3. État actuel` décrit la réalité du repo aujourd’hui.
- Les sections `6` à `11` décrivent les chantiers encore utiles.
- Les checklists servent de suivi opérationnel.

### 2.2 Règle de mise à jour

À chaque vraie avancée structurante :

1. cocher les items terminés ;
2. dater la revue dans le tableau de suivi ;
3. noter la prochaine vérification ;
4. mettre à jour les risques si le périmètre a changé.

### 2.3 Convention de suivi

- `[x]` = terminé
- `[ ]` = à faire ou encore incomplet

Pour un chantier partiellement avancé, on laisse le titre du chantier non coché et on coche seulement les sous-items terminés.

---

## 3. État actuel du repo

## 3.1 Frontend

Le frontend n’est plus dans une phase “pré-migration”.

La réalité actuelle est la suivante :

- Vue 3 + Pinia sont intégrés ;
- les racines Vue existent (`js/vue/App.vue`, `js/vue/GeneratedFeedApp.vue`, `js/vue/GlobalRuntime.vue`) ;
- le panel principal, la grille, la sidebar, le feed généré, les menus contextuels et les hosts viewer sont désormais portés par Vue ;
- `AssetsManagerPanel.js`, `panelState.js`, plusieurs factories DOM legacy et plusieurs shims historiques ont déjà été supprimés ;
- le runtime panel a commencé à être redécoupé via :
  - `js/features/panel/panelRuntime.js`
  - `js/features/panel/panelBootstrap.js`
  - `js/features/panel/panelFiltersInit.js`
  - `js/features/panel/panelGridEventBindings.js`
  - `js/features/panel/panelPinnedFolders.js`
  - `js/features/panel/panelSettingsSync.js`
  - `js/features/panel/panelSimilarSearch.js`
  - `js/features/panel/panelContextMenuExtraActions.js`
  - `js/features/panel/panelRuntimeRefs.js`

Conséquence :

- l’ancien chantier “faire la migration frontend Vue” n’est plus un objectif central ;
- le chantier utile maintenant est **consolider, tester, simplifier et nettoyer l’après-migration Vue**.

Le document détaillé de migration frontend reste :

- `docs/VUE_MIGRATION_PLAN.md`

Ce plan complet ne le remplace pas comme historique détaillé. Il le **recontextualise** dans la roadmap globale.

## 3.2 Backend

Le backend est déjà structuré par grands domaines :

- `mjr_am_backend/features/*`
- `mjr_am_backend/routes/*`
- `mjr_am_backend/adapters/*`

Mais plusieurs points restent denses :

- `mjr_am_backend/routes/handlers/assets_impl.py`
- `mjr_am_backend/routes/handlers/search_impl.py`
- `mjr_am_backend/routes/registry.py`
- certaines zones security / observability / bootstrap

La couche DB est **déjà amorcée** côté découpage, avec par exemple :

- `mjr_am_backend/adapters/db/sqlite_facade.py`
- `mjr_am_backend/adapters/db/connection_pool.py`
- `mjr_am_backend/adapters/db/transaction_manager.py`
- `mjr_am_backend/adapters/db/db_recovery.py`
- `mjr_am_backend/adapters/db/schema.py`

Donc ici aussi, l’objectif n’est plus “inventer le split”, mais **finir et clarifier le split existant**.

## 3.3 Gouvernance dépendances et docs

Le repo possède déjà :

- `requirements.txt`
- `requirements-vector.txt`
- `pyproject.toml`

Mais à ce stade, la gouvernance n’est pas encore totalement verrouillée :

- `requirements-dev.txt` n’est pas encore officialisé ;
- la politique de vérité unique n’est pas encore formalisée dans une doc dédiée ;
- le README, les scripts d’installation et les règles de contribution doivent rester alignés.

---

## 4. Vision cible mise à jour

La cible n’est plus seulement “migrer vers Vue”.

La cible est un projet avec des frontières lisibles entre :

```text
1. frontend_vue_shell
   - racines Vue
   - stores Pinia
   - composants UI
   - composables de lifecycle

2. frontend_runtime_services
   - viewer runtime
   - DnD
   - intégration ComfyUI
   - bridges event/runtime

3. asset_catalog
   - indexation
   - recherche
   - métadonnées
   - lecture des assets

4. asset_operations
   - rename
   - delete
   - download
   - tags / rating
   - opérations fichiers

5. ai_enrichment
   - vector search
   - captions
   - alignment
   - clustering / smart collections

6. platform_runtime
   - bootstrap
   - registry de routes
   - security
   - observability
   - dependency policy
```

---

## 5. Tableau de suivi global

| Chantier | État actuel | Dernière revue | Prochaine vérification | Remarques |
|---|---|---|---|---|
| Migration Vue des surfaces UI majeures | Fait | 2026-04-03 | À clôturer dans ce plan | Voir `docs/VUE_MIGRATION_PLAN.md` |
| Consolidation frontend post-Vue | En cours | 2026-04-03 | Après chaque lot de tests / cleanup | Nouveau chantier frontend prioritaire |
| Découpage DB | En cours | 2026-04-03 | Après chaque extraction interne | Split déjà amorcé |
| Split handlers assets/search | À lancer proprement | 2026-04-03 | Lors du prochain lot backend | `assets_impl.py` reste trop dense |
| Registry / middlewares / bootstrap routes | À faire | 2026-04-03 | Après handlers | `registry.py` reste central |
| Clarification security / observability | À faire | 2026-04-03 | Après registry | Important mais pas premier |
| Gouvernance dépendances / docs / ADR | Partiellement fait | 2026-04-03 | À la prochaine passe docs | `requirements-vector.txt` existe, politique incomplète |

---

## 6. Checklist maître

## 6.1 Déjà réalisé

- [x] Intégrer Vue 3 + Pinia dans le frontend
- [x] Migrer les racines UI principales vers Vue
- [x] Migrer grille, sidebar, feed et menus contextuels vers Vue
- [x] Introduire `panelRuntime.js` en remplacement du shell panel historique
- [x] Supprimer plusieurs factories DOM et shims legacy devenus obsolètes
- [x] Introduire des tests ciblés pour les composants Vue viewer critiques

## 6.2 À terminer côté frontend

- [ ] Formaliser la fin de la migration frontend dans le plan maître
- [ ] Réduire les bridges legacy encore nécessaires mais trop implicites
- [ ] Continuer le découpage du runtime viewer impératif derrière la façade Vue
- [ ] Continuer le découpage des helpers DnD / runtime encore trop transverses
- [ ] Étendre la couverture de tests Vue sur les composants les plus critiques
- [ ] Clarifier par écrit ce qui reste impératif “par design” et ce qui reste impératif “temporairement”

## 6.3 À terminer côté backend

- [ ] Finaliser le découpage interne de la couche DB
- [ ] Découper `assets_impl.py` en routes et services par responsabilité
- [ ] Découper `search_impl.py` si la densité continue d’augmenter
- [ ] Amincir `routes/registry.py`
- [ ] Revoir la séparation security / auth / tokens / proxies / rate-limit
- [ ] Clarifier les politiques de fallback et de mode strict dev

## 6.4 À terminer côté gouvernance technique

- [ ] Officialiser la politique dépendances runtime / vector / dev
- [ ] Ajouter `requirements-dev.txt`
- [ ] Ajouter une doc `docs/DEPENDENCY_POLICY.md`
- [ ] Mettre à jour README / docs d’installation / contribution
- [ ] Ajouter ou compléter les ADR liées au nouveau frontend Vue et au runtime

---

## 7. Chantier A — Frontend post-Vue

**Nouveau statut : ce chantier remplace l’ancien “migrer le frontend vers Vue”.**

## 7.1 Objectif

Consolider l’architecture maintenant que la majorité des surfaces UI sont déjà en Vue.

Le but n’est plus de déplacer des templates DOM vers Vue.
Le but est de :

- sécuriser les interfaces runtime entre Vue et les services ;
- réduire les bridges historiques encore nécessaires ;
- améliorer la testabilité ;
- rendre explicites les zones impératives conservées volontairement.

## 7.2 Périmètre actuel

### Fait

- panel shell Vue
- grid shell Vue
- sidebar Vue
- generated feed Vue
- context menus Vue
- viewer runtime hosts Vue
- stores Pinia

### Encore sensibles

- `js/features/viewer/*`
- `js/components/Viewer_impl.js`
- `js/features/dnd/*`
- certains contrôleurs et bridges panel encore nécessaires

## 7.3 Cible

Séparer clairement :

```text
frontend_vue_shell/
  composants, stores, composables, mounting

frontend_runtime_services/
  viewer, DnD, runtime event bridges, ComfyUI integration
```

## 7.4 Actions recommandées

- [ ] Documenter ce qui reste impératif par design dans le viewer
- [ ] Identifier les bridges frontend encore provisoires
- [ ] Regrouper les conventions de lifecycle Vue/runtime dans une doc courte
- [ ] Étendre les tests Vue pour :
  - [ ] hosts viewer
  - [ ] menus contextuels viewer/grid
  - [ ] composants panel critiques
  - [ ] régressions de teardown / listeners
- [ ] Réduire progressivement les accès implicites à `window.*`
- [ ] Continuer le split du runtime panel si des responsabilités restent mélangées

## 7.5 Critères d’acceptation

- Vue reste propriétaire des surfaces UI majeures
- les services impératifs restants sont explicitement justifiés
- les points de montage et de teardown sont testés
- les régressions de lifecycle deviennent faciles à localiser

---

## 8. Chantier B — Couche DB

## 8.1 Objectif

Terminer le découpage de la couche SQLite en gardant une façade publique stable.

## 8.2 Réalité actuelle

Le split a déjà commencé.
Le plan doit donc évoluer de “imaginer la structure” vers “stabiliser la structure existante”.

## 8.3 Structure cible mise à jour

```text
mjr_am_backend/adapters/db/
  sqlite_facade.py
  sqlite.py
  connection_pool.py
  transaction_manager.py
  db_recovery.py
  schema.py
  ...
```

Le nom exact des fichiers peut évoluer, mais la règle reste la même :

- la façade publique reste fine ;
- les responsabilités lourdes sont déplacées dans des modules dédiés.

## 8.4 Checklist

- [ ] Cartographier ce qui reste encore concentré dans `sqlite_facade.py` et `sqlite.py`
- [ ] Isoler clairement l’exécution SQL
- [ ] Isoler clairement les transactions
- [ ] Isoler clairement recovery / reset / heal
- [ ] Isoler clairement les helpers de guards SQL
- [ ] Isoler clairement le spécifique Windows si encore mélangé ailleurs
- [ ] Ajouter des tests ciblés par sous-zone extraite

## 8.5 Critères d’acceptation

- l’API DB publique ne casse pas le reste du backend
- la complexité des gros fichiers diminue vraiment
- les tests se localisent mieux par responsabilité

---

## 9. Chantier C — Split handlers assets et search

## 9.1 Objectif

Éliminer les handlers trop centraux et pousser la logique métier dans `features/*`.

## 9.2 Fichiers prioritaires

- `mjr_am_backend/routes/handlers/assets_impl.py`
- `mjr_am_backend/routes/handlers/assets.py`
- `mjr_am_backend/routes/handlers/search_impl.py`

## 9.3 Structure cible

```text
mjr_am_backend/routes/handlers/
  asset_mutation_routes.py
  asset_rating_tags_routes.py
  asset_download_routes.py
  asset_docs_routes.py
  search_query_routes.py
  search_similarity_routes.py
  asset_ops_shared.py
```

Et côté métier :

```text
mjr_am_backend/features/assets/
  request_contexts.py
  rename_service.py
  delete_service.py
  rating_tags_service.py
  download_service.py
  path_resolution_service.py
```

## 9.4 Checklist

- [ ] Séparer les routes rename / delete / batch delete
- [ ] Séparer les routes rating / tags / autocomplete
- [ ] Séparer les routes download / download-clean / preview
- [ ] Sortir les helpers HTTP partagés dans un module commun léger
- [ ] Déplacer les validations request → context dans `features/assets`
- [ ] Réduire les accès DB/filesystem directs dans les handlers
- [ ] Réévaluer `search_impl.py` avec la même logique si la densité le justifie

## 9.5 Critères d’acceptation

- les handlers restent des façades HTTP fines
- la logique métier est testable indépendamment des routes
- les routes publiques côté client restent compatibles

---

## 10. Chantier D — Registry, bootstrap routes, security, observability

## 10.1 Objectif

Réduire les modules “centres nerveux” backend.

## 10.2 Fichiers sensibles

- `mjr_am_backend/routes/registry.py`
- `mjr_am_backend/routes/core/security.py`
- `mjr_am_backend/observability.py`
- `__init__.py`

## 10.3 Cible

### Registry

- `registry.py` doit orchestrer, pas tout faire
- l’ordre des modules de routes doit devenir plus déclaratif

### Security

Séparer autant que possible :

- auth policy
- token auth
- rate limiting
- trusted proxies
- CSRF
- bridges ComfyUI éventuels

### Observability

Séparer autant que possible :

- request context
- logging rate-limité
- asyncio exception bridge

### Bootstrap

Conserver `__init__.py` compatible ComfyUI, mais réduire les side effects directs.

## 10.4 Checklist

- [ ] Extraire un catalogue de routes déclaratif
- [ ] Sortir les middlewares / bridges PromptServer si encore entremêlés
- [ ] Revoir le bootstrap import-side-effects
- [ ] Clarifier les sous-zones de `security.py`
- [ ] Clarifier les sous-zones de `observability.py`
- [ ] Ajouter un mode strict dev plus explicite pour les fallbacks critiques

## 10.5 Critères d’acceptation

- ordre d’enregistrement lisible
- security et observability plus auditables
- bootstrap moins risqué à modifier

---

## 11. Chantier E — Dépendances, docs, ADR, qualité

## 11.1 Objectif

Éviter la double vérité et laisser une documentation alignée avec le code actuel.

## 11.2 Réalité actuelle

Déjà présent :

- `requirements.txt`
- `requirements-vector.txt`
- `pyproject.toml`
- `README.md`
- `docs/adr/`

Encore manquant ou à formaliser :

- `requirements-dev.txt`
- politique explicite de gouvernance des dépendances
- ADR dédiées au nouveau frontend Vue et à l’après-migration

## 11.3 Checklist

- [ ] Décider officiellement si `requirements.txt` reste la source de vérité runtime
- [ ] Créer `requirements-dev.txt`
- [ ] Créer `docs/DEPENDENCY_POLICY.md`
- [ ] Mettre README à jour avec la structure frontend Vue actuelle
- [ ] Ajouter une ADR “Frontend Vue ownership / runtime bridges”
- [ ] Ajouter une ADR “Panel runtime split”
- [ ] Vérifier l’alignement docs / scripts / install / qualité
- [ ] Relever progressivement les exigences de tests ciblés sur les fichiers changés

## 11.4 Critères d’acceptation

- plus de divergence évidente entre docs et repo
- onboarding plus simple
- les règles de dépendances ne reposent plus sur de l’implicite

---

## 12. Phasage recommandé mis à jour

## Phase 0 — Alignement documentaire et suivi

- [ ] Valider ce plan maître
- [ ] Rattacher explicitement le frontend terminé à `docs/VUE_MIGRATION_PLAN.md`
- [ ] Ajouter la politique de dépendances
- [ ] Créer le tableau de suivi vivant dans les revues

## Phase 1 — Consolidation frontend post-Vue

- [ ] Étendre les tests Vue critiques
- [ ] Stabiliser les contrats runtime Vue ↔ services
- [ ] Documenter l’impératif conservé par design
- [ ] Poursuivre la simplification viewer / DnD / panel runtime

## Phase 2 — Finalisation du split DB

- [ ] Terminer les extractions internes DB
- [ ] Réduire la façade à un rôle d’assemblage
- [ ] Ajouter les tests par sous-module

## Phase 3 — Split handlers assets/search

- [ ] Réduire `assets_impl.py`
- [ ] Réduire `search_impl.py` si nécessaire
- [ ] Déplacer la logique métier côté `features/*`

## Phase 4 — Registry / security / observability

- [ ] Amincir `registry.py`
- [ ] Rendre les modules de routes plus déclaratifs
- [ ] Clarifier `security.py`
- [ ] Clarifier `observability.py`

## Phase 5 — Nettoyage final et durcissement

- [ ] Nettoyer les reliquats legacy restants
- [ ] Rehausser progressivement les checks qualité
- [ ] Stabiliser les conventions de maintenance

---

## 13. Checklist de revue à cocher à chaque lot

À utiliser comme mini check de suivi après chaque PR de refacto.

- [ ] Le comportement public est inchangé ou explicitement documenté
- [ ] Les tests ciblés du périmètre passent
- [ ] Le build frontend passe si du JS/Vue a changé
- [ ] Les modules déplacés ont encore un point d’entrée lisible
- [ ] Les fichiers legacy supprimables ont été supprimés ou explicitement conservés
- [ ] La doc a été mise à jour si l’architecture visible a changé
- [ ] Les fallbacks nouveaux sont intentionnels et pas accidentels
- [ ] Le chantier a été recoché / redaté dans ce plan

---

## 14. Ordre de priorité recommandé maintenant

Ordre conseillé à partir de l’état actuel :

1. Consolider l’après-migration Vue
2. Finaliser le split DB
3. Découper `assets_impl.py` et les handlers voisins
4. Amincir `registry.py`
5. Clarifier security / observability
6. Verrouiller dépendances / ADR / docs / qualité

Ordre à éviter maintenant :

1. repartir dans une “nouvelle migration frontend” générique
2. ouvrir un gros redesign produit au milieu du cleanup structurel
3. mélanger ajout de features majeures et gros split backend dans les mêmes PR

---

## 15. Définition du succès

Le refacto sera considéré comme réussi si :

- le frontend Vue reste la surface principale sans régression de lifecycle ;
- les services impératifs restants sont volontairement identifiés ;
- les gros handlers backend cessent d’être des points de congestion ;
- la couche DB devient plus lisible à maintenir ;
- `registry.py`, `security.py` et `observability.py` cessent d’être des zones “à ne pas toucher” ;
- les docs et dépendances reflètent enfin le repo réel ;
- le suivi du chantier est faisable simplement en cochant ce document.

---

## 16. Résumé opérationnel condensé

```text
FAIT
- migration majeure du frontend vers Vue
- panel/grid/sidebar/feed/context menus/hôtes viewer en Vue
- premier split du runtime panel

PROCHAINES PRIORITÉS
- consolider l’après-migration Vue
- étendre les tests Vue critiques
- finaliser le split DB
- casser assets_impl.py par responsabilités
- aminci registry/security/observability
- formaliser dependency policy + ADR + suivi
```

---

## 17. Note finale

Le plan précédent restait pertinent sur le fond, mais il n’était plus aligné avec la réalité du code :

- il traitait encore le frontend Vue comme un chantier principalement à faire ;
- il ne reflétait pas le découpage déjà effectué dans le panel runtime ;
- il ne proposait pas de mécanisme simple de suivi par vérification.

La nouvelle version conserve l’intention d’origine, mais la remet dans le bon ordre :

- **ce qui est déjà fait est reconnu comme fait** ;
- **ce qui reste à faire est reformulé comme travail utile maintenant** ;
- **le document peut servir directement de support de suivi grâce aux checklists**.
