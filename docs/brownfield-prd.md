# jieba_fast_dat Brownfield Enhancement PRD

## Intro Project Analysis and Context

### Existing Project Overview

**Current Project State:**
`jieba_fast_dat` 專案是一個用於**文本分詞**的函式庫，主要使用 **Python 和 C** 語言開發。其核心功能是將文本切分成詞語，以支援自然語言處理任務。

### Enhancement Scope Definition

**Enhancement Type:** 主要功能修改
**Enhancement Description:** 透過將原版 Jieba 的記憶體 Trie 替換為 Double-Array Trie (DAT)，並利用 mmap 載入預先建構的 DAT 快取文件，以大幅降低詞典載入記憶體和提升載入速度。同時，移除執行時動態添加使用者詞的功能，改為透過修改使用者詞典文件並重新運行程式來更新。
**Impact Assessment:** 重大影響 (substantial existing code changes)

### Goals and Background Context

**Goals:**
*   提升 `jieba_fast_dat` 的整體效能。
*   大幅降低 `jieba_fast_dat` 在詞典載入時的 RAM 使用量。
*   縮短詞典建構與載入的時間。

**Background Context:**
`jieba_fast_dat` 專案目前面臨詞典建構和載入時間過長的問題，這影響了函式庫的初始化速度和資源效率。為了解決這個痛點，本次增強功能旨在透過引入 Double-Array Trie (DAT) 和 mmap 載入機制，從根本上優化詞典的處理方式，從而提升效能並降低記憶體佔用，使其更適用於對資源敏感的應用場景。

### Change Log

| Date       | Version | Description         | Author |
| :--------- | :------ | :------------------ | :----- |
| 2025-10-14 | 1.0     | Initial brownfield PRD | PM     |

## Requirements

### Functional

1.  **FR1**: 系統應使用 Double-Array Trie (DAT) 作為詞典儲存的核心資料結構。
2.  **FR2**: 系統應利用記憶體映射 (mmap) 機制載入預先建構的 DAT 快取文件，以實現快速初始化。
3.  **FR3**: 系統應具備自動偵測使用者詞典文件變更的能力，並在程式重新運行時自動重建 DAT 快取。

### Non Functional

1.  **NFR1**: 詞典載入時的 RAM 使用量應顯著低於現有實作。
2.  **NFR2**: 詞典載入時間應顯著快於現有實作。
3.  **NFR3**: 系統將不再支援執行時動態添加使用者詞 (例如 `add_word` / `InsertUserWord` 等功能將被移除)。

### Compatibility Requirements

1.  **CR1**: 現有的文本分詞 API 介面（除了動態添加使用者詞的功能外）應保持不變，以確保對現有使用者的影響最小化。
2.  **CR2**: 詞典文件格式應與新的 DAT 結構相容，或提供明確且易於操作的詞典遷移或轉換方案。
3.  **CR3**: 文本分詞的整體行為和準確性應與現有實作保持一致，確保不會引入新的分詞錯誤或降低分詞品質。

## Technical Constraints and Integration Requirements

### Existing Technology Stack

**Languages**: Python, C
**Frameworks**: (假設可能使用 Cython 或 CFFI 進行 Python-C 整合，但目前未明確指定)
**Database**: 無 (假設 `jieba_fast_dat` 作為分詞函式庫，核心功能不直接依賴資料庫)
**Infrastructure**: 無 (假設為標準 Python 套件，無特定基礎設施要求)
**External Dependencies**: (除了 Python 和 C 語言本身，假設無其他核心外部依賴，但可能包含用於編譯 C 程式碼的工具鏈)

### Integration Approach

**Database Integration Strategy**: 無 (本次增強功能不涉及資料庫整合)
**API Integration Strategy**:
*   **Python API**: 新的 DAT 實作應透過現有的 Python 介面提供分詞功能，確保對外 API 保持一致性 (除了移除動態詞典功能)。
*   **C 模組整合**: DAT 的核心邏輯應在 C 語言層面實現，並透過 Python-C 介面 (如 Cython 或 CFFI) 與 Python 部分無縫整合。
**Frontend Integration Strategy**: 無 (本次增強功能不涉及前端整合)
**Testing Integration Strategy**: 應整合到現有的測試框架中，確保新舊功能都能得到充分測試。

### Code Organization and Standards

**File Structure Approach**: 新增的 DAT 相關 C 程式碼和 Python 綁定程式碼應遵循現有專案的檔案結構和模組劃分，保持一致性。
**Naming Conventions**: 新增的程式碼應遵循現有的 Python 和 C 程式碼命名規範。
**Coding Standards**: 新增的程式碼應遵循現有的 Python PEP 8 和 C 語言編碼規範。
**Documentation Standards**: 新增或修改的程式碼應包含必要的註釋和文件，解釋其功能、用途和任何重要考量。

### Deployment and Operations

**Build Process Integration**: 應將 DAT 快取文件的建構過程整合到現有的專案建構流程中，確保自動化。
**Deployment Strategy**: 應遵循現有的 Python 套件部署策略，確保新版本能順利發布。
**Monitoring and Logging**: 應利用現有的監控和日誌系統，追蹤新實作的效能和潛在問題。
**Configuration Management**: 應利用現有的配置管理方式，處理詞典路徑、快取文件位置等配置。

### Risk Assessment and Mitigation

**Technical Risks**:
*   **DAT 實作複雜性**: DAT 結構的正確實現和高效能綁定可能存在技術挑戰。
*   **Python-C 介面效能瓶頸**: Python 和 C 之間資料傳遞和函數呼叫可能引入新的效能瓶頸。
*   **快取重建機制**: 自動偵測詞典變更並重建快取機制的穩定性和可靠性。
**Integration Risks**:
*   **與現有程式碼的相容性**: 新的 DAT 實作可能與現有程式碼庫的其他部分產生衝突。
*   **API 變更影響**: 移除動態詞典功能可能對依賴此功能的現有使用者造成影響。
**Deployment Risks**:
*   **建構流程複雜化**: 引入 DAT 快取建構可能增加建構流程的複雜性。
*   **快取文件管理**: 快取文件的儲存、更新和清理可能需要額外的部署考量。
**Mitigation Strategies**:
*   **逐步實作與測試**: 將 DAT 替換分解為小步驟，並在每個階段進行充分的單元測試和整合測試。
*   **效能基準測試**: 在開發過程中持續進行效能基準測試，確保達到預期的效能提升和記憶體降低目標。
*   **明確的 API 變更通知**: 對於移除的動態詞典功能，應在發布說明中明確指出，並提供替代方案。
*   **自動化建構與測試**: 確保建構和測試流程完全自動化，減少人為錯誤。

## Epic and Story Structure

### Epic Approach

**Epic Structure Decision**: 單一綜合史詩，因為此增強功能是一個連貫的工作單元，專注於優化核心詞典機制。

### Epic 1: 詞典效能與記憶體優化

**Epic Goal**: 透過引入 Double-Array Trie (DAT) 和 mmap 載入機制，全面優化 `jieba_fast_dat` 的詞典儲存、載入效能和記憶體使用，同時調整動態詞典功能以適應新的架構。

**Integration Requirements**: 新的 DAT 實作必須與現有的 Python/C 程式碼庫無縫整合，確保現有分詞 API 的穩定性（除了動態詞典功能），並遵循現有的建構、部署和測試流程。

#### Story 1.1 DAT 核心結構與建構

**As a** 開發者,
**I want** 實現 Double-Array Trie (DAT) 的核心資料結構和建構邏輯,
**so that** 為詞典的高效儲存和查詢奠定基礎。

**Acceptance Criteria**
1.  DAT 結構能夠正確儲存詞典中的所有詞語。
2.  提供一個 C 語言實現的 DAT 建構器，能夠從詞典文件生成 DAT 結構。
3.  DAT 建構過程的記憶體使用和時間複雜度符合預期優化目標。
4.  提供單元測試驗證 DAT 結構的正確性和建構器的功能。

**Integration Verification**
1.  IV1: 驗證 DAT 建構器能夠處理現有詞典文件格式。
2.  IV2: 驗證 DAT 結構能夠被 Python 程式碼正確調用和操作。

#### Story 1.2 mmap 載入與詞典替換

**As a** 開發者,
**I want** 實現 DAT 快取文件的 mmap 載入機制，並將現有詞典替換為 DAT 實作,
**so that** 詞典載入速度和記憶體使用得到顯著提升。

**Acceptance Criteria**
1.  系統能夠利用 mmap 載入預先建構的 DAT 快取文件。
2.  現有分詞邏輯能夠成功切換到使用 DAT 進行詞語查詢。
3.  詞典載入時間和記憶體使用量符合 NFR1 和 NFR2 的要求。
4.  提供單元測試和效能測試，驗證 mmap 載入和 DAT 查詢的效能。

**Integration Verification**
1.  IV1: 驗證現有分詞 API 在切換到 DAT 後，功能和準確性保持不變 (CR3)。
2.  IV2: 驗證詞典載入過程與現有建構和部署流程整合。

#### Story 1.3 動態詞典功能移除與快取重建

**As a** 開發者,
**I want** 移除動態添加使用者詞的功能，並實現使用者詞典文件變更時自動重建 DAT 快取機制,
**so that** 系統行為與新的詞典管理方式保持一致。

**Acceptance Criteria**
1.  `add_word` / `InsertUserWord` 等動態添加使用者詞的功能從公共 API 中移除。
2.  系統能夠偵測使用者詞典文件的變更。
3.  當使用者詞典文件變更時，系統在重新啟動時能夠自動觸發 DAT 快取重建。
4.  提供單元測試驗證快取重建機制的正確性。

**Integration Verification**
1.  IV1: 驗證移除動態詞典功能後，不會對其他現有 API 造成負面影響 (CR1)。
2.  IV2: 驗證快取重建機制與現有檔案系統操作和建構流程整合。
